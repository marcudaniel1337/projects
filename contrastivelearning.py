import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models

# =============================================
# Configurations
# =============================================
class Config:
    batch_size = 256           # Number of samples per batch
    epochs = 100               # Total number of training epochs
    lr = 3e-4                  # Learning rate for optimizer
    temperature = 0.5          # Temperature parameter for NT-Xent loss
    projection_dim = 128       # Output dimension of the projection head
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = 4            # DataLoader workers
    image_size = 32            # Input image dimension (e.g., CIFAR-10)

# =============================================
# Data Augmentation for SimCLR
# =============================================
# Two random transformations for each image to create contrastive pairs
simclr_transforms = transforms.Compose([
    transforms.RandomResizedCrop(Config.image_size),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class SimCLRDataset(Dataset):
    """
    Custom Dataset for SimCLR which returns two augmented views of the same image
    """
    def __init__(self, root, train=True, transform=None):
        self.dataset = datasets.CIFAR10(root, train=train, download=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]  # We ignore labels for self-supervised learning
        # Apply two independent augmentations
        xi = self.transform(image)
        xj = self.transform(image)
        return xi, xj

# =============================================
# SimCLR Model Definition
# =============================================
class ProjectionHead(nn.Module):
    """
    2-layer MLP as projection head: maps representation to space where contrastive loss is applied
    """
    def __init__(self, input_dim, hidden_dim=512, output_dim=Config.projection_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class SimCLR(nn.Module):
    """
    SimCLR model combining a CNN encoder and projection head
    """
    def __init__(self, base_model, projection_dim):
        super().__init__()
        # Load pre-defined CNN (e.g., ResNet18) and remove final classification layer
        self.encoder = base_model(pretrained=False)
        num_ftrs = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.projection_head = ProjectionHead(num_ftrs, hidden_dim=num_ftrs, output_dim=projection_dim)

    def forward(self, x):
        h = self.encoder(x)         # Extract representation
        z = self.projection_head(h) # Project to contrastive space
        return h, z

# =============================================
# Contrastive Loss (NT-Xent)
# =============================================
def nt_xent_loss(z_i, z_j, temperature):
    """
    Compute normalized temperature-scaled cross entropy loss
    z_i, z_j: embeddings of two augmented sets (batch_size x dim)
    """
    batch_size = z_i.shape[0]
    # Normalize projections
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)

    # Concatenate into one tensor: 2N samples
    z = torch.cat([z_i, z_j], dim=0)
    # Compute similarity matrix
    sim_matrix = torch.matmul(z, z.T) / temperature

    # Create labels: positive pairs are diagonal offsets
    labels = torch.arange(batch_size, device=Config.device)
    labels = torch.cat([labels, labels], dim=0)

    # Mask self-similarities
    mask = (~torch.eye(2 * batch_size, 2 * batch_size, dtype=bool)).to(Config.device)
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

    # Compute cross-entropy loss
    loss = F.cross_entropy(sim_matrix, labels)
    return loss

# =============================================
# Training Loop
# =============================================
def train(model, loader, optimizer, epoch):
    """
    One epoch of training SimCLR
    """
    model.train()
    total_loss = 0.0
    for batch_idx, (xi, xj) in enumerate(loader):
        xi, xj = xi.to(Config.device), xj.to(Config.device)
        # Forward pass
        _, zi = model(xi)
        _, zj = model(xj)
        # Contrastive loss
        loss = nt_xent_loss(zi, zj, Config.temperature)
        # Backpropagation update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (batch_idx + 1) % 20 == 0:
            avg = total_loss / (batch_idx + 1)
            print(f"Epoch [{epoch}/{Config.epochs}] Batch [{batch_idx+1}/{len(loader)}] Loss: {avg:.4f}")

    return total_loss / len(loader)

# =============================================
# Main Execution
# =============================================
if __name__ == '__main__':
    # Prepare dataset and dataloader
    dataset = SimCLRDataset(root='./data', train=True, transform=simclr_transforms)
    loader = DataLoader(dataset, batch_size=Config.batch_size,
                        shuffle=True, num_workers=Config.num_workers, drop_last=True)

    # Initialize SimCLR model
    model = SimCLR(models.resnet18, projection_dim=Config.projection_dim).to(Config.device)
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)

    # Training
    for epoch in range(1, Config.epochs + 1):
        avg_loss = train(model, loader, optimizer, epoch)
        print(f"--> Epoch {epoch} completed. Average Loss: {avg_loss:.4f}\n")

    # After training, the encoder can be fine-tuned or used as a feature extractor
