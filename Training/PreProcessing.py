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
from tkinter.filedialog import askdirectory
from skimage import io, measure, exposure
from Model.ResNet50_model import resnet50
from torch import nn, optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


def create_output_folder(input_folder, output_folder_name):
    """
    Create an output folder if it does not exist.
    :param input_folder: Path to the input folder
    :param output_folder_name: Name of the output folder
    :return: Full path of the output folder
    """
    output_folder = os.path.join(input_folder, output_folder_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created {output_folder_name} output folder: {output_folder}")
    else:
        print(f"{output_folder_name} output folder already exists")
    return output_folder


def extract_single_cells(input_folder, num_channels, input_folder_2):
    """
    Extract single cell images from the input folder.
    :param input_folder: List of input folders
    :param num_channels: Number of channels
    :param input_folder_2: List of folders to store single cell images
    """
    for j in range(num_channels):
        output_folder = create_output_folder(input_folder[j], 'SingleCell')
        input_folder_2.append(output_folder)

        files_tif = sorted([f for f in os.listdir(input_folder[j]) if f.lower().endswith(('.tif', '.tiff'))])
        files_png = sorted([f for f in os.listdir(input_folder[3]) if f.lower().endswith('.png')])
        image_counter = 0
        num_files = len(files_tif)

        for i in range(num_files):
            try:
                tif_file = files_tif[i]
                png_file = files_png[i]

                tif_path = os.path.join(input_folder[j], tif_file)
                png_path = os.path.join(input_folder[3], png_file)

                imager = io.imread(tif_path)
                mask = io.imread(png_path)

                image = np.zeros_like(imager, dtype=np.uint16)
                image[mask > 0] = imager[mask > 0]

                labeled_mask = measure.label(mask > 0)
                regions = measure.regionprops(labeled_mask, intensity_image=image)

                for region in regions:
                    min_row, min_col, max_row, max_col = region.bbox
                    min_row = max(min_row, 0)
                    min_col = max(min_col, 0)
                    max_row = min(max_row, imager.shape[0])
                    max_col = min(max_col, imager.shape[1])

                    xWidth = max_col - min_col - 1
                    yHeight = max_row - min_row - 1

                    if xWidth < 0 or yHeight < 0:
                        print(f"Invalid bounding box, region number: {region.label}, file: {tif_file}. Skipping this region.")
                        continue

                    imageCut = np.zeros((yHeight + 1, xWidth + 1), dtype=np.uint16)
                    coords = region.coords
                    rows_adjusted = coords[:, 0] - min_row
                    cols_adjusted = coords[:, 1] - min_col
                    imageCut[rows_adjusted, cols_adjusted] = imager[coords[:, 0], coords[:, 1]]

                    image_counter += 1
                    filename = f'DR_{image_counter}.tif'
                    output_path = os.path.join(output_folder, filename)
                    io.imsave(output_path, imageCut)

            except Exception as e:
                print(f"Error processing file {tif_file}: {e}")

            per = (j * num_files + i + 1) / (num_channels * num_files) * 100
            sys.stdout.write(f"\rProcessing progress {per:.2f}% ")

    sys.stdout.write("\n")
    sys.stdout.write(f"Successfully obtained single cell images")
    sys.stdout.write("\n")


def pad_single_cells(input_folder, num_channels, input_folder_2, input_folder_3):
    """
    Pad single cell images with zeros.
    :param input_folder: List of input folders
    :param num_channels: Number of channels
    :param input_folder_2: List of folders to store single cell images
    :param input_folder_3: List of folders to store padded single cell images
    """
    for j in range(num_channels):
        image_dir = input_folder_2[j]
        output_folder = create_output_folder(input_folder[j], 'SingleCell_padding')
        input_folder_3.append(output_folder)

        max_width = 120
        max_height = 120
        images = []
        image_files = []

        for filename in os.listdir(image_dir):
            if filename.endswith(('.tif', '.png', '.jpg', '.jpeg')):
                img_path = os.path.join(image_dir, filename)
                try:
                    with Image.open(img_path) as img:
                        if img.width < max_width and img.height < max_height:
                            images.append(img.copy())
                            image_files.append(filename)
                        else:
                            print(filename)
                except Exception as e:
                    print(f"Error opening file {filename}: {e}")

        for img, filename in zip(images, image_files):
            try:
                img_tensor = torch.from_numpy(np.array(img))
                diffY = max_height - img_tensor.size(0) + 30
                diffX = max_width - img_tensor.size(1) + 30
                padding = [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
                padded_img = F.pad(img_tensor, padding)

                padded_img = padded_img.numpy().astype(np.uint16)
                padded_img = Image.fromarray(padded_img)

                output_filename = filename.split('.')[0] + '.tif'
                output_path = os.path.join(output_folder, output_filename)
                padded_img.save(output_path, format='TIFF')
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    sys.stdout.write(f"Edge zero-padding completed")
    sys.stdout.write("\n")


if __name__ == "__main__":
    # Assume num_channels and input_folder are already defined
    num_channels = 3  # Example value, need to be modified according to actual situation
    input_folder = ['path/to/channel1', 'path/to/channel2', 'path/to/channel3', 'path/to/mask']  # Example value, need to be modified according to actual situation

    input_folder_2 = []
    input_folder_3 = []

    extract_single_cells(input_folder, num_channels, input_folder_2)
    pad_single_cells(input_folder, num_channels, input_folder_2, input_folder_3)