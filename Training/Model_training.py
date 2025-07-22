import os
import torch
import random
import psutil
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, classification_report
from Model.ResNet50_model import resnet50

# --- Hardware Configuration ---
def log_hardware_info():
    """Log available hardware resources"""
    # CPU information
    cpu_cores = os.cpu_count()
    print(f"Available CPU cores: {cpu_cores}")
    
    # RAM information
    memory = psutil.virtual_memory()
    print(f"Total RAM: {memory.total / (1024**3):.2f} GB")
    print(f"Available RAM: {memory.available / (1024**3):.2f} GB")
    
    # GPU information
    if torch.cuda.is_available():
        print(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name} | Memory: {props.total_memory/1e9:.2f} GB")
    else:
        print("No CUDA-capable devices available")

# Set device configuration
def configure_device():
    """Configure computation device and log specifications"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        capability = torch.cuda.get_device_capability(device)
        props = torch.cuda.get_device_properties(device)
        total_cores = props.multi_processor_count * 128  # 128 cores per SM
        print(f"Compute Capability: {capability[0]}.{capability[1]}")
        print(f"Total CUDA cores: {total_cores}")
    
    return device

# --- Data Augmentation Classes ---
class RandomCropResize(transforms.Resize):
    """Randomly crops and resizes image while maintaining aspect ratio"""
    def __init__(self, original_size=120, scale_range=(0.92, 1.00), interpolation=Image.BILINEAR):
        super().__init__(size=(original_size, original_size), interpolation=interpolation)
        self.original_size = original_size
        self.scale_min, self.scale_max = scale_range

    def __call__(self, img):
        scale = random.uniform(self.scale_min, self.scale_max)
        crop_size = int(self.original_size * scale)
        img = transforms.CenterCrop(crop_size)(img)
        return super().__call__(img)

# --- Dataset Classes ---
class BacteriaDataset(Dataset):
    """Custom dataset for bacterial cell images"""
    def __init__(self, root_dir, label, transform=None):
        self.root_dir = root_dir
        self.label = label
        self.transform = transform
        self.image_paths = [
            os.path.join(root_dir, f) 
            for f in os.listdir(root_dir) 
            if f.endswith('.tif')
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = np.array(Image.open(img_path))
        
        # Normalize and convert to 8-bit
        img = (img / np.max(img) * 255).astype(np.uint8)
        img = Image.fromarray(img, mode='L')
        
        if self.transform:
            img = self.transform(img)
            
        return img, self.label

# --- Utility Functions ---
def save_as_tiff(array, filename):
    """Save numpy array as 16-bit TIFF image"""
    if array.dtype != np.uint16:
        array = (array * 65535).astype(np.uint16)
    Image.fromarray(array, mode='I;16').save(filename, format='TIFF')

def amplify_dataset(dataset, output_dir, num_augmentations=20):
    """Generate augmented versions of dataset images"""
    os.makedirs(output_dir, exist_ok=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    for i, (img, _) in enumerate(loader):
        for j in range(num_augmentations):
            # Apply augmentation transforms
            aug_img = train_transform(img).numpy().squeeze()
            save_path = os.path.join(
                output_dir, 
                f"aug_{i*num_augmentations + j + 1}.tif"
            )
            save_as_tiff(aug_img, save_path)

def create_data_loaders(datasets, test_ratio=0.2, batch_size=48, num_workers=8):
    """Create train and test loaders from multiple datasets"""
    train_subsets, test_subsets = [], []
    
    for dataset in datasets:
        total_size = len(dataset)
        test_size = int(total_size * test_ratio)
        train_size = total_size - test_size
        train_sub, test_sub = random_split(dataset, [train_size, test_size])
        train_subsets.append(train_sub)
        test_subsets.append(test_sub)
    
    train_data = ConcatDataset(train_subsets)
    test_data = ConcatDataset(test_subsets)
    
    # Collate functions for dynamic augmentation
    def train_collate(batch):
        imgs = [train_transform(img) for img, _ in batch]
        labels = [label for _, label in batch]
        return torch.stack(imgs), torch.tensor(labels)
    
    def test_collate(batch):
        imgs = [test_transform(img) for img, _ in batch]
        labels = [label for _, label in batch]
        return torch.stack(imgs), torch.tensor(labels)
    
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        collate_fn=train_collate, num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False,
        collate_fn=test_collate, num_workers=num_workers
    )
    
    return train_loader, test_loader

# --- Model Configuration ---
def modify_resnet(model, in_channels=1, num_classes=5):
    """Adapt ResNet50 for grayscale input and custom classes"""
    # Modify first convolution layer for grayscale input
    original_conv1 = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels, original_conv1.out_channels,
        kernel_size=original_conv1.kernel_size,
        stride=original_conv1.stride,
        padding=original_conv1.padding,
        bias=False
    )
    
    # Modify final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

# --- Training and Evaluation ---
def train_epoch(model, loader, criterion, optimizer, device, writer, epoch):
    """Train model for one epoch"""
    model.train()
    running_loss = 0.0
    
    for step, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Log every 10 batches
        if (step + 1) % 10 == 0:
            avg_loss = running_loss / (step + 1)
            writer.add_scalar(
                'train_loss/batch', 
                avg_loss, 
                epoch * len(loader) + step
            )
    
    return running_loss / len(loader)

def evaluate(model, loader, device, writer, epoch):
    """Evaluate model on test set"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Log every 10 batches
            if (i + 1) % 10 == 0:
                batch_acc = correct / total
                writer.add_scalar(
                    'test_acc/batch', 
                    batch_acc, 
                    epoch * len(loader) + i
                )
    
    accuracy = correct / total
    return accuracy

def generate_classification_report(model, loader, device, class_names):
    """Generate confusion matrix and classification report"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    cr = classification_report(
        all_labels, all_preds, 
        target_names=class_names, 
        output_dict=True
    )
    
    # Create DataFrames for better visualization
    cm_df = pd.DataFrame(
        cm, 
        index=[f"True_{name}" for name in class_names],
        columns=[f"Pred_{name}" for name in class_names]
    )
    
    cr_df = pd.DataFrame(cr).transpose()
    return cm_df, cr_df

# --- Main Execution ---
if __name__ == "__main__":
    # Hardware setup
    log_hardware_info()
    device = configure_device()
    
    # Configuration parameters
    DATA_ROOT = "/mnt/e/wsl_share/Machine_learning/DR/DR_latest/HJ/Cellwalltest_20241219"
    CLASS_NAMES = ['stage1', 'stage2', 'stage3', 'stage4', 'stage5']
    NUM_CLASSES = len(CLASS_NAMES)
    AUGMENTATION_FACTORS = [7, 6, 8, 50, 250]
    LOG_DIR = "runs/cellwall_classification"
    MODEL_SAVE_PATH = "models/resnet50_cellwall.pth"
    
    # Create data augmentation transforms
    train_transform = transforms.Compose([
        transforms.RandomRotation(180),
        RandomCropResize(original_size=150),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])
    
    # Create base transform for initial loading
    base_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Create test transforms
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(180),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])
    
    # Step 1: Dataset amplification
    original_datasets = [
        BacteriaDataset(
            os.path.join(DATA_ROOT, f"stage{i+1}"), 
            i, 
            transform=base_transform
        ) 
        for i in range(NUM_CLASSES)
    ]
    
    for i, (ds, factor) in enumerate(zip(original_datasets, AUGMENTATION_FACTORS)):
        output_dir = os.path.join(DATA_ROOT, f"stage{i+1}_amplified")
        amplify_dataset(ds, output_dir, num_augmentations=factor)
    
    # Step 2: Prepare augmented datasets
    augmented_datasets = [
        BacteriaDataset(
            os.path.join(DATA_ROOT, f"stage{i+1}_amplified"), 
            i
        ) 
        for i in range(NUM_CLASSES)
    ]
    
    # Step 3: Create data loaders
    train_loader, test_loader = create_data_loaders(
        augmented_datasets, 
        test_ratio=0.2,
        batch_size=48,
        num_workers=8
    )
    
    # Step 4: Initialize model
    model = resnet50(pretrained=True)
    model = modify_resnet(model, in_channels=1, num_classes=NUM_CLASSES)
    model = model.to(device)
    
    # Step 5: Training setup
    writer = SummaryWriter(LOG_DIR)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    
    # Add model graph to TensorBoard
    dummy_input = torch.randn(1, 1, 224, 224).to(device)
    writer.add_graph(model, dummy_input)
    
    # Step 6: Training loop
    best_accuracy = 0.0
    NUM_EPOCHS = 30
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*30}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        # Training phase
        train_loss = train_epoch(
            model, train_loader, 
            criterion, optimizer, 
            device, writer, epoch
        )
        
        # Evaluation phase
        test_acc = evaluate(
            model, test_loader, 
            device, writer, epoch
        )
        
        print(f"Train Loss: {train_loss:.4f} | Test Acc: {test_acc:.4f}")
        
        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Saved new best model with accuracy: {best_accuracy:.4f}")
    
    # Step 7: Final evaluation
    cm_df, cr_df = generate_classification_report(
        model, test_loader, device, CLASS_NAMES
    )
    
    # Save evaluation metrics
    cm_df.to_csv(os.path.join(LOG_DIR, "confusion_matrix.csv"))
    cr_df.to_csv(os.path.join(LOG_DIR, "classification_report.csv"))
    
    print("\nTraining completed!")
    writer.close()