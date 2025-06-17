# Author: oxh
# Last Modified: 2024-12-10

import os
import re
import sys
import time
import glob
import torch
import warnings
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from tkinter import Tk
from tkinter import simpledialog
from tkinter.filedialog import askdirectory
from skimage import io, measure, exposure
from Model.ResNet50 import resnet50
from torch import nn, optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

class SingleCellClassifier:
    """
    A class for classifying single cells using ResNet50 model.
    This class handles the entire workflow from data preparation to model inference and result classification.
    """
    
    def __init__(self):
        """Initialize the classifier with default parameters."""
        self.num_channels = 3
        self.ch_used2classify = 2  # Default channel for classification: Channel2
        self.num_class = 5
        self.class_names = ['stage1', 'stage2', 'stage3', 'stage4', 'stage5']
        self.weights_path = None
        self.input_folder = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def setup_parameters(self):
        """Set up initial parameters for the classification process."""
        print("=== Step 1: Setting up parameters ===")
        
        # Ask user for number of channels
        root = Tk()
        root.withdraw()
        self.num_channels = simpledialog.askinteger(
            "Input", "Enter number of channels (3 or 4):",
            minvalue=3, maxvalue=4
        )
        root.destroy()
        
        if self.num_channels is None:
            self.num_channels = 3
        print(f"Number of channels set to: {self.num_channels}")
        
        # Ask user for classification type (cell wall or cell membrane)
        root = Tk()
        root.withdraw()
        parameter = simpledialog.askstring(
            "Input", "cellwall (cw) or cellmembrane (cm)?"
        )
        root.destroy()
        
        if parameter in ["cellwall", "cw", "CellWall", "CW"]:
            self.weights_path = './Model/trained_parameters/resnet50ForDR_5features(CellWall).pth'
        elif parameter in ["cellmembrane", "cm", "CellMembrane", "CM"]:
            self.weights_path = './Model/trained_parameters/resnet50ForDR_5features(CellMembrane).pth'
        else:
            print("Invalid input. Please enter 'cw' for cell wall or 'cm' for cell membrane.")
            sys.exit(0)
            
        print(f"Selected weights path: {self.weights_path}")
    
    def select_input_folder(self):
        """Select the input folder containing image data."""
        print("=== Step 2: Selecting input folder ===")
        
        # Disable specific warnings
        warnings.simplefilter("ignore", UserWarning)
        
        # Ask user to select input folder
        root = Tk()
        root.withdraw()
        input_folder0 = askdirectory(title='Select input folder')
        root.destroy()
        
        self.input_folder = sorted(glob.glob(os.path.join(input_folder0, '*/')))
        
        if not self.input_folder:
            print("No subfolders found in the selected directory.")
            sys.exit(0)
            
        # Get image files from the first and last subfolders
        files_tif = sorted([f for f in os.listdir(self.input_folder[0]) if f.lower().endswith(('.tif', '.tiff'))])
        files_png = sorted([f for f in os.listdir(self.input_folder[-1]) if f.lower().endswith('.png')])
        
        # Check if the number of .tif and .png files match
        if len(files_tif) != len(files_png):
            print("Mismatch between the number of .tif and .png files. Please check the input folder.")
            sys.exit(0)
        else:
            print(f"Found {len(files_tif)} pairs of .tif and .png files. Continuing...")
            
        # Create class-specific output folders for each channel
        for num in range(self.num_channels):
            channel_folder = self.input_folder[num]
            for class_name in self.class_names:
                class_folder = os.path.join(channel_folder, class_name)
                if not os.path.exists(class_folder):
                    os.makedirs(class_folder)
                    
        return files_png
    
    def extract_single_cells(self, files_png):
        """
        Extract single cells from the input images using provided masks.
        
        Args:
            files_tif (list): List of .tif image files.
            files_png (list): List of .png mask files.
            
        Returns:
            list: List of output folders containing single cell images.
        """
        print("=== Step 3: Extracting single cell images ===")
        
        input_folder_2 = []
        
        for j in range(self.num_channels):
            output_folder_name = 'SingleCell'
            output_folder = os.path.join(self.input_folder[j], output_folder_name)
            
            # Create output folder if it doesn't exist
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
                print(f"Created output folder: {output_folder}")
            else:
                print(f"Output folder already exists: {output_folder}")
                
            input_folder_2.append(output_folder)
            
            # Get .tif files for this channel
            channel_files_tif = sorted([f for f in os.listdir(self.input_folder[j]) if f.lower().endswith(('.tif', '.tiff'))])
            num_files = len(channel_files_tif)
            image_counter = 0
            
            # Process each image-mask pair
            for i in range(num_files):
                tif_file = channel_files_tif[i]
                png_file = files_png[i]
                
                # Read images
                tif_path = os.path.join(self.input_folder[j], tif_file)
                png_path = os.path.join(self.input_folder[-1], png_file)
                
                imager = io.imread(tif_path)
                mask = io.imread(png_path)
                
                # Create output image with only masked regions
                image = np.zeros_like(imager, dtype=np.uint16)
                image[mask > 0] = imager[mask > 0]
                
                # Label regions in the mask
                labeled_mask = measure.label(mask > 0)
                regions = measure.regionprops(labeled_mask, intensity_image=image)
                
                # Extract and save each cell region
                for region in regions:
                    min_row, min_col, max_row, max_col = region.bbox
                    
                    # Ensure valid bounding box
                    min_row = max(min_row, 0)
                    min_col = max(min_col, 0)
                    max_row = min(max_row, imager.shape[0])
                    max_col = min(max_col, imager.shape[1])
                    
                    xWidth = max_col - min_col - 1
                    yHeight = max_row - min_row - 1
                    
                    if xWidth < 0 or yHeight < 0:
                        print(f"Invalid bounding box for region {region.label} in file {tif_file}. Skipping...")
                        continue
                    
                    # Extract cell region
                    imageCut = np.zeros((yHeight + 1, xWidth + 1), dtype=np.uint16)
                    coords = region.coords
                    rows_adjusted = coords[:, 0] - min_row
                    cols_adjusted = coords[:, 1] - min_col
                    imageCut[rows_adjusted, cols_adjusted] = imager[coords[:, 0], coords[:, 1]]
                    
                    # Save the extracted cell image
                    image_counter += 1
                    filename = f'DR_{image_counter}.tif'
                    output_path = os.path.join(output_folder, filename)
                    io.imsave(output_path, imageCut)
                
                # Print progress
                per = (j*num_files + i + 1) / (self.num_channels * num_files) * 100
                sys.stdout.write(f"\rProcessing progress: {per:.2f}% ")
                
            sys.stdout.write("\n")
            
        print(f"Successfully extracted single cell images.")
        return input_folder_2
    
    def pad_single_cells(self, input_folder_2):
        """
        Pad single cell images to a uniform size with zeros.
        
        Args:
            input_folder_2 (list): List of input folders containing single cell images.
            
        Returns:
            list: List of output folders containing padded single cell images.
        """
        print("=== Step 4: Padding single cell images ===")
        
        input_folder_3 = []
        
        for j in range(self.num_channels):
            image_dir = input_folder_2[j]
            output_folder_name = 'SingleCell_padding'
            output_folder = os.path.join(self.input_folder[j], output_folder_name)
            
            # Create output folder if it doesn't exist
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
                print(f"Created padding output folder: {output_folder}")
            else:
                print(f"Padding output folder already exists: {output_folder}")
                
            input_folder_3.append(output_folder)
            
            # Process each image
            max_width = 120
            max_height = 120
            processed_count = 0
            
            for filename in os.listdir(image_dir):
                if filename.endswith(('.tif', '.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(image_dir, filename)
                    with Image.open(img_path) as img:
                        if img.width < max_width and img.height < max_height:
                            # Convert image to tensor and pad
                            img_tensor = torch.from_numpy(np.array(img))
                            diffY = max_height - img_tensor.size(0) + 30
                            diffX = max_width - img_tensor.size(1) + 30
                            padding = [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
                            padded_img = F.pad(img_tensor, padding)
                            
                            # Convert back to image and save
                            padded_img = padded_img.numpy().astype(np.uint16)
                            padded_img = Image.fromarray(padded_img)
                            
                            output_filename = filename.split('.')[0] + '.tif'
                            output_path = os.path.join(output_folder, output_filename)
                            padded_img.save(output_path, format='TIFF')
                            
                            processed_count += 1
            
            print(f"Padded {processed_count} images for channel {j+1}")
            
        print("Padding completed.")
        return input_folder_3
    
    def classify_cells(self, input_folder_3):
        """
        Classify cells using the ResNet50 model.
        
        Args:
            input_folder_3 (list): List of input folders containing padded single cell images.
        """
        print("=== Step 5: Classifying cells using ResNet50 ===")
        
        class ImageFolderDataset(Dataset):
            """Custom dataset class for loading images from a folder."""
            def __init__(self, folder_path, transform=None):
                self.image_paths = [os.path.join(folder_path, filename) 
                                    for filename in os.listdir(folder_path) 
                                    if filename.endswith('.tif')]
                self.transform = transform
                
            def __len__(self):
                return len(self.image_paths)
                
            def __getitem__(self, idx):
                image_path = self.image_paths[idx]
                image = np.array(Image.open(image_path))
                image = (image/np.max(image) * 255).astype(np.uint8)
                image = Image.fromarray(image, mode='L')
                
                if self.transform:
                    image = self.transform(image)
                    
                return image, image_path
        
        # Define transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Load dataset and create data loader
        folder_path = input_folder_3[self.ch_used2classify - 1]
        dataset = ImageFolderDataset(folder_path, transform=transform)
        data_loader = DataLoader(dataset, batch_size=50, shuffle=False)
        
        print(f"Using device: {self.device}")
        
        # Initialize and load model
        net = resnet50(num_classes=self.num_class)
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        net.load_state_dict(torch.load(self.weights_path, map_location=self.device))
        net.to(self.device)
        net.eval()
        
        # Perform inference
        results = []
        
        with torch.no_grad():
            for images, paths in data_loader:
                images = images.to(self.device)
                outputs = net(images)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                top_p, top_classes = torch.topk(probabilities, 2, dim=1)
                
                for path, probs, classes in zip(paths, top_p, top_classes):
                    img_name = os.path.basename(path)
                    top1_class = self.class_names[classes[0].item()]
                    
                    result = {
                        'Image': img_name,
                        'Top1_Class': top1_class,
                        'Top1_Prob': probs[0].item(),
                        'Top2_Class': self.class_names[classes[1].item()],
                        'Top2_Prob': probs[1].item()
                    }
                    results.append(result)
                    
                    # Copy images to corresponding class folders
                    c2_s = path
                    c1_s = c2_s.replace('C2', 'C1')
                    c3_s = c2_s.replace('C2', 'C3')
                    
                    c1_t = os.path.join(self.input_folder[0], top1_class, img_name)
                    c2_t = os.path.join(self.input_folder[1], top1_class, img_name)
                    c3_t = os.path.join(self.input_folder[2], top1_class, img_name)
                    
                    shutil.copy(c1_s, c1_t)
                    shutil.copy(c2_s, c2_t)
                    shutil.copy(c3_s, c3_t)
                    
                    if self.num_channels == 4:
                        c4_s = c2_s.replace('C2', 'C4')
                        c4_t = os.path.join(self.input_folder[3], top1_class, img_name)
                        shutil.copy(c4_s, c4_t)
        
        # Sort results and save to CSV
        results.sort(key=lambda x: int(re.search(r'\d+', x['Image']).group()))
        input_folder0 = os.path.dirname(self.input_folder[0])
        output_csv = os.path.join(input_folder0, 'classification_results.csv')
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv, index=False)
        
        print(f"Classification completed. Results saved to {output_csv}")
        print("Check the classification results in the input directory and class-specific folders.")

def main():
    """Main function to execute the single cell classification workflow."""
    print("=== Single Cell Classification Workflow ===")
    
    classifier = SingleCellClassifier()
    classifier.setup_parameters()
    files_tif, files_png = classifier.select_input_folder()
    input_folder_2 = classifier.extract_single_cells(files_tif, files_png)
    input_folder_3 = classifier.pad_single_cells(input_folder_2)
    classifier.classify_cells(input_folder_3)
    
    print("=== Workflow completed successfully ===")

if __name__ == "__main__":
    main()