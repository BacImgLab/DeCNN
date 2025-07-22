import torch
from torch import nn

class Bottleneck(nn.Module):
    """
    Bottleneck residual block for ResNet.
    
    This block implements the bottleneck architecture with three convolutional layers:
    1x1 reduction, 3x3 convolution, and 1x1 expansion.
    It includes batch normalization and ReLU activation after each convolution.
    """
    # Expansion factor for the output channels
    expansion = 4  # Class attribute shared by all Bottleneck instances
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        Initialize the Bottleneck block.
        
        Args:
            inplanes (int): Number of input channels.
            planes (int): Number of channels in the intermediate layers.
            stride (int, optional): Stride for the first convolution. Defaults to 1.
            downsample (nn.Module, optional): Downsample layer for the residual connection. Defaults to None.
        """
        super(Bottleneck, self).__init__()
        
        # 1x1 convolution layer
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # 3x3 convolution layer
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # 1x1 convolution layer for expanding channels
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Downsample layer for the residual connection
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        """
        Forward pass of the Bottleneck block.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor after processing through the block.
        """
        # Save the residual for later addition
        residual = x
        
        # First convolution block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second convolution block
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        # Third convolution block (no ReLU after this)
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Apply downsample to the residual if necessary
        if self.downsample is not None:
            residual = self.downsample(x)
            
        # Add the residual to the output
        out += residual
        out = self.relu(out)
        
        return out

class ResNet(nn.Module):
    """
    Residual Network (ResNet) architecture.
    
    This implementation follows the design principles described in the paper:
    "Deep Residual Learning for Image Recognition" by He et al. (2015).
    """
    def __init__(self, block, layers, num_classes=1000):
        """
        Initialize the ResNet model.
        
        Args:
            block (nn.Module): Block type to use (e.g., Bottleneck).
            layers (list): List of integers specifying the number of blocks in each layer.
            num_classes (int, optional): Number of output classes. Defaults to 1000.
        """
        # Number of input channels for the first convolutional layer
        self.inplanes = 64
        super(ResNet, self).__init__()
        
        # Model configuration
        self.block = block
        self.layers = layers
        
        # Stem layers (initial convolutional block)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Final classification layers
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Create a residual layer consisting of multiple blocks.
        
        Args:
            block (nn.Module): Block type to use.
            planes (int): Number of channels in the intermediate layers.
            blocks (int): Number of blocks in this layer.
            stride (int, optional): Stride for the first block. Defaults to 1.
            
        Returns:
            nn.Sequential: A sequential module containing the residual blocks.
        """
        downsample = None
        
        # Create downsample layer if necessary (stride != 1 or channel mismatch)
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
            
        # First block is a Conv Block (may have downsampling)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        
        # Remaining blocks are Identity Blocks
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass of the ResNet model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 3, height, width].
            
        Returns:
            torch.Tensor: Output tensor after processing through the network.
        """
        # Stem layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Classification layers
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

def resnet50(num_classes=1000):
    """
    Create a ResNet-50 model.
    
    Args:
        num_classes (int, optional): Number of output classes. Defaults to 1000.
        
    Returns:
        ResNet: A ResNet-50 model instance.
    """
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)

if __name__ == '__main__':
    model = resnet50()
    # summary(model, (3, 224, 224), device='cpu')