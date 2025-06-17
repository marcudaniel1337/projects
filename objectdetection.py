import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
import yaml

# Let's start by creating a custom dataset class for our object detection task
# This will handle loading images and their corresponding bounding box annotations
class CustomDetectionDataset(Dataset):
    """
    Custom dataset class for object detection.
    This handles loading images and their bounding box annotations in YOLO format.
    """
    
    def __init__(self, images_dir, labels_dir, img_size=640, transform=None):
        """
        Initialize our custom dataset
        
        Args:
            images_dir: Directory containing our training images
            labels_dir: Directory containing YOLO format label files (.txt)
            img_size: Size to resize images to (YOLOv5 typically uses 640x640)
            transform: Any additional transforms to apply to images
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.transform = transform
        
        # Get all image files - we'll support common formats
        self.image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            self.image_files.extend([f for f in os.listdir(images_dir) if f.lower().endswith(ext)])
        
        print(f"Found {len(self.image_files)} images in dataset")
    
    def __len__(self):
        """Return the total number of images in our dataset"""
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Get a single item from our dataset
        This is where the magic happens - we load the image and its labels
        """
        # Load the image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # OpenCV loads images in BGR format, but we want RGB
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Store original dimensions - we'll need these for scaling bounding boxes
        orig_h, orig_w = image.shape[:2]
        
        # Resize image to our target size (keeping aspect ratio would be better, but let's keep it simple)
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        # Convert to tensor and normalize to [0,1] range
        image = torch.from_numpy(image).float() / 255.0
        image = image.permute(2, 0, 1)  # Convert from HWC to CHW format (PyTorch expects this)
        
        # Now let's load the corresponding labels
        label_name = img_name.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
        label_path = os.path.join(self.labels_dir, label_name)
        
        boxes = []
        labels = []
        
        # Check if label file exists (some images might not have annotations)
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    # YOLO format: class_id center_x center_y width height (all normalized 0-1)
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        center_x = float(parts[1])
                        center_y = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Convert from YOLO format to absolute coordinates
                        # Scale to our resized image dimensions
                        x1 = (center_x - width/2) * self.img_size
                        y1 = (center_y - height/2) * self.img_size
                        x2 = (center_x + width/2) * self.img_size
                        y2 = (center_y + height/2) * self.img_size
                        
                        boxes.append([x1, y1, x2, y2])
                        labels.append(class_id)
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        
        return image, boxes, labels

# Now let's implement a simplified YOLOv5 architecture
# The real YOLOv5 is quite complex, so we'll create a simplified but functional version
class YOLOv5(nn.Module):
    """
    Simplified YOLOv5 implementation
    This is a basic version that captures the essence of YOLO architecture
    """
    
    def __init__(self, num_classes=80, img_size=640):
        super(YOLOv5, self).__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        
        # YOLOv5 uses three different scales for detection (similar to FPN)
        # We'll use three anchor sizes for simplicity
        self.num_anchors = 3
        
        # Backbone - we'll use a simple CNN backbone (in real YOLOv5, this would be CSPDarknet)
        self.backbone = nn.Sequential(
            # First convolution block
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            
            # Downsample and increase channels
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            
            # More convolution blocks
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
        )
        
        # Detection head - this predicts bounding boxes and class probabilities
        # For each anchor, we predict: x, y, w, h, objectness, class_probs
        # So total output channels = num_anchors * (5 + num_classes)
        self.detection_head = nn.Conv2d(512, self.num_anchors * (5 + num_classes), kernel_size=1)
        
        # Initialize weights properly (very important for training stability)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights - this is crucial for good training"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Xavier initialization works well for conv layers
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network
        """
        # Extract features using backbone
        features = self.backbone(x)
        
        # Get predictions from detection head
        predictions = self.detection_head(features)
        
        # Reshape predictions for easier processing
        # Original shape: [batch, anchors*(5+classes), grid_h, grid_w]
        # We want: [batch, anchors, grid_h, grid_w, 5+classes]
        batch_size = predictions.size(0)
        grid_size = predictions.size(2)  # Assuming square grid
        
        predictions = predictions.view(batch_size, self.num_anchors, 
                                     5 + self.num_classes, grid_size, grid_size)
        predictions = predictions.permute(0, 1, 3, 4, 2)  # Move channels to last dimension
        
        return predictions

class YOLOLoss(nn.Module):
    """
    YOLO Loss function
    This is where we calculate how wrong our predictions are
    """
    
    def __init__(self, num_classes=80, img_size=640):
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # Loss weights - these are hyperparameters that balance different loss components
        self.lambda_coord = 5.0  # Weight for coordinate loss
        self.lambda_obj = 1.0    # Weight for objectness loss
        self.lambda_noobj = 0.5  # Weight for no-object loss
        self.lambda_class = 1.0  # Weight for classification loss
    
    def forward(self, predictions, targets):
        """
        Calculate YOLO loss
        This is simplified - real YOLO loss is more complex with anchor matching
        """
        # For simplicity, we'll use a basic loss calculation
        # In practice, you'd need proper anchor matching and more sophisticated loss computation
        
        batch_size = predictions.size(0)
        grid_size = predictions.size(2)
        
        # Extract different parts of the prediction
        pred_boxes = predictions[..., :4]      # x, y, w, h
        pred_conf = predictions[..., 4]        # objectness confidence
        pred_class = predictions[..., 5:]      # class probabilities
        
        # For this simplified version, we'll just use MSE loss
        # In a real implementation, you'd have complex target assignment
        loss = torch.tensor(0.0, requires_grad=True)
        
        # This is a placeholder - real YOLO loss requires proper target formatting
        # and anchor matching which is quite complex
        
        return loss

def train_model(model, train_loader, val_loader, num_epochs=100, lr=0.001):
    """
    Training function - this is where our model learns to detect objects
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = model.to(device)
    
    # Adam optimizer works well for most deep learning tasks
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler - reduces LR when loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # Loss function
    criterion = YOLOLoss(num_classes=model.num_classes)
    
    # Keep track of losses for plotting
    train_losses = []
    val_losses = []
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, boxes, labels) in enumerate(train_loader):
            images = images.to(device)
            # For simplicity, we're not using boxes and labels in this basic version
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(images)
            
            # Calculate loss (simplified)
            loss = criterion(predictions, None)  # Placeholder
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            train_loss += loss.item()
            
            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, boxes, labels in val_loader:
                images = images.to(device)
                predictions = model(images)
                loss = criterion(predictions, None)  # Placeholder
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_losses.append(avg_val_loss)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Save model checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, f'yolov5_checkpoint_epoch_{epoch+1}.pth')
    
    return train_losses, val_losses

def detect_objects(model, image_path, conf_threshold=0.5, device='cpu'):
    """
    Perform object detection on a single image
    This is where we use our trained model to find objects in new images
    """
    model.eval()
    model = model.to(device)
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    original_image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    image_resized = cv2.resize(image, (model.img_size, model.img_size))
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image_resized).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # Add batch dimension
    image_tensor = image_tensor.to(device)
    
    # Get predictions
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # Process predictions (this is simplified)
    # In a real implementation, you'd need Non-Maximum Suppression (NMS)
    # and proper coordinate conversion
    
    detections = []
    
    # This is a placeholder for actual detection processing
    # Real implementation would involve:
    # 1. Converting grid predictions to absolute coordinates
    # 2. Applying confidence threshold
    # 3. Non-Maximum Suppression to remove duplicate detections
    # 4. Scaling coordinates back to original image size
    
    return detections, original_image

def visualize_detections(image, detections, class_names=None):
    """
    Visualize detection results on the image
    This helps us see what our model detected
    """
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    # Draw bounding boxes for each detection
    for detection in detections:
        x1, y1, x2, y2, conf, class_id = detection
        
        # Create rectangle patch
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        # Add label
        label = f'Class {class_id}: {conf:.2f}'
        if class_names and class_id < len(class_names):
            label = f'{class_names[class_id]}: {conf:.2f}'
        
        ax.text(x1, y1-10, label, fontsize=12, color='red', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_title('Object Detection Results')
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def create_sample_config():
    """
    Create a sample configuration file for our custom dataset
    This helps organize our training parameters
    """
    config = {
        'dataset': {
            'train_images': 'data/train/images',
            'train_labels': 'data/train/labels',
            'val_images': 'data/val/images', 
            'val_labels': 'data/val/labels',
            'num_classes': 2,  # Adjust based on your dataset
            'class_names': ['person', 'car']  # Add your class names here
        },
        'training': {
            'batch_size': 16,
            'learning_rate': 0.001,
            'num_epochs': 100,
            'img_size': 640,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        },
        'model': {
            'model_name': 'custom_yolov5',
            'pretrained': False
        }
    }
    
    # Save configuration
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("Sample configuration saved to config.yaml")
    return config

# Example usage and main training script
def main():
    """
    Main function to orchestrate the entire training process
    This is like the conductor of our object detection orchestra
    """
    
    print("=== Custom YOLOv5 Object Detection Training ===")
    
    # Create sample configuration
    config = create_sample_config()
    
    # Set up dataset paths (adjust these to your actual data)
    train_images_dir = "data/train/images"
    train_labels_dir = "data/train/labels"
    val_images_dir = "data/val/images"
    val_labels_dir = "data/val/labels"
    
    # Check if data directories exist
    if not os.path.exists(train_images_dir):
        print(f"Warning: Training images directory '{train_images_dir}' not found!")
        print("Please organize your data in the following structure:")
        print("data/")
        print("├── train/")
        print("│   ├── images/")
        print("│   └── labels/")
        print("└── val/")
        print("    ├── images/")
        print("    └── labels/")
        return
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = CustomDetectionDataset(train_images_dir, train_labels_dir, 
                                         img_size=config['training']['img_size'])
    
    val_dataset = CustomDetectionDataset(val_images_dir, val_labels_dir,
                                       img_size=config['training']['img_size']) if os.path.exists(val_images_dir) else None
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, 
                             batch_size=config['training']['batch_size'], 
                             shuffle=True, 
                             num_workers=4)
    
    val_loader = DataLoader(val_dataset, 
                           batch_size=config['training']['batch_size'], 
                           shuffle=False, 
                           num_workers=4) if val_dataset else None
    
    # Create model
    print("Creating YOLOv5 model...")
    model = YOLOv5(num_classes=config['dataset']['num_classes'],
                   img_size=config['training']['img_size'])
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {total_params:,} total parameters ({trainable_params:,} trainable)")
    
    # Start training
    print("Starting training process...")
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['num_epochs'],
        lr=config['training']['learning_rate']
    )
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log Scale)')
    plt.title('Training Loss (Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'train_losses': train_losses,
        'val_losses': val_losses
    }, 'final_yolov5_model.pth')
    
    print("Training completed! Model saved as 'final_yolov5_model.pth'")
    
    # Example of using the trained model for detection
    # if os.path.exists('sample_image.jpg'):
    #     print("Running detection on sample image...")
    #     detections, image = detect_objects(model, 'sample_image.jpg')
    #     visualize_detections(image, detections, config['dataset']['class_names'])

if __name__ == "__main__":
    # This runs when the script is executed directly
    main()

# Additional utility functions you might find useful

def convert_coco_to_yolo(coco_json_path, output_dir):
    """
    Convert COCO format annotations to YOLO format
    This is handy if you have COCO format data
    """
    # This is a placeholder - implement based on your needs
    pass

def augment_dataset(images_dir, labels_dir, output_dir, augmentation_factor=3):
    """
    Apply data augmentation to increase dataset size
    More data usually means better model performance
    """
    # This would implement various augmentations like:
    # - Random flips
    # - Random rotations
    # - Color jittering
    # - Random crops
    # - Mixup/Cutmix
    pass

def evaluate_model(model, test_loader, class_names):
    """
    Evaluate model performance using standard metrics
    Like mAP (mean Average Precision)
    """
    # This would calculate:
    # - Precision
    # - Recall  
    # - mAP@0.5
    # - mAP@0.5:0.95
    pass

# Remember: This is a simplified implementation of YOLOv5
# The real YOLOv5 has many more components like:
# - CSPDarknet backbone
# - PANet neck
# - Multiple detection scales
# - Anchor-based detection
# - Advanced loss functions
# - Extensive data augmentation
# - Model ensembling
# - And much more!

print("YOLOv5 Custom Object Detection implementation loaded!")
print("Don't forget to prepare your dataset in YOLO format before training!")
