import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import random


# Fix random seed for reproducibility
def fix_random_seed(seed=42):
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.benchmark = False  # Disable CuDNN benchmarking for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)  # Python hash seed

# Set the random seed
fix_random_seed(42)

class UTKFaceDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        self.valid_files = []

        # Filter out invalid files
        for filename in self.image_files:
            try:
                age, gender, ethnicity, _ = filename.split('_')
                age = int(age)
                gender = int(gender)
                ethnicity = int(ethnicity)
                self.valid_files.append(filename)
            except ValueError:
                print(f"Skipping invalid file: {filename}")
         
        # self.valid_files = self.valid_files[:100]

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.valid_files[idx])
        image = Image.open(img_name).convert('RGB')

        # Parse annotations from filename
        filename = self.valid_files[idx]
        age, gender, ethnicity, _ = filename.split('_')
        age = int(age)
        gender = int(gender)
        ethnicity = int(ethnicity)

        # Categorize age into Child (0-17), Adult (18-64), Senior (65+)
        if age < 18:
            age_category = 0  # Child
        elif 18 <= age <= 64:
            age_category = 1  # Adult
        else:
            age_category = 2  # Senior

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Return image and annotations
        return image, torch.tensor(age_category, dtype=torch.long), torch.tensor(gender, dtype=torch.long), torch.tensor(ethnicity, dtype=torch.long)   

# Training transformations with augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])


# Validation/Test transformations (no augmentation)
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets with respective transformations
train_dataset = UTKFaceDataset(root_dir='../datasets/UTKFace/crop_part1/', transform=train_transform)
test_dataset = UTKFaceDataset(root_dir='../datasets/UTKFace/crop_part1/', transform=test_transform)

# Split dataset into train and test
train_size = int(0.8 * len(train_dataset))
test_size = len(train_dataset) - train_size

train_dataset, _ = torch.utils.data.random_split(train_dataset, [train_size, test_size])
_, test_dataset = torch.utils.data.random_split(test_dataset, [train_size, test_size])

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

# Define the model
class MultiTaskMobileNetV2(nn.Module):
    def __init__(self, num_age_classes, num_gender_classes, num_ethnicity_classes):
        super(MultiTaskMobileNetV2, self).__init__()
        self.backbone = models.mobilenet_v2(pretrained=True)
        self.backbone.classifier[1] = nn.Identity()  # Remove the final classification layer

        # Add task-specific heads
        self.age_head = nn.Linear(1280, num_age_classes)  # 3 classes for age
        self.gender_head = nn.Linear(1280, num_gender_classes)
        self.ethnicity_head = nn.Linear(1280, num_ethnicity_classes)

    def forward(self, x):
        features = self.backbone(x)
        age = self.age_head(features)
        gender = self.gender_head(features)
        ethnicity = self.ethnicity_head(features)
        return age, gender, ethnicity

# Initialize the model
model = MultiTaskMobileNetV2(num_age_classes=3, num_gender_classes=2, num_ethnicity_classes=5)

# Wrap the model with DataParallel for multi-GPU training
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

# Move model to GPU(s)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define loss functions and optimizer
criterion_age = nn.CrossEntropyLoss()
criterion_gender = nn.CrossEntropyLoss()
criterion_ethnicity = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Initialize TensorBoard
writer = SummaryWriter('runs/utkface_experiment')

# Training function
def train(model, dataloader, optimizer, criterion_age, criterion_gender, criterion_ethnicity, device, epoch):
    model.train()
    running_loss = 0.0
    for i, (images, age_labels, gender_labels, ethnicity_labels) in enumerate(dataloader):
        images, age_labels, gender_labels, ethnicity_labels = images.to(device), age_labels.to(device), gender_labels.to(device), ethnicity_labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        age_pred, gender_pred, ethnicity_pred = model(images)

        # Compute losses
        loss_age = criterion_age(age_pred, age_labels)
        loss_gender = criterion_gender(gender_pred, gender_labels)
        loss_ethnicity = criterion_ethnicity(ethnicity_pred, ethnicity_labels)
        loss = loss_age + loss_gender + loss_ethnicity

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Log batch loss to TensorBoard
        writer.add_scalar('Training Loss (Batch)', loss.item(), epoch * len(dataloader) + i)

    # Log epoch loss to TensorBoard
    epoch_loss = running_loss / len(dataloader)
    writer.add_scalar('Training Loss (Epoch)', epoch_loss, epoch)

    return epoch_loss

# Evaluation function
def evaluate(model, dataloader, criterion_age, criterion_gender, criterion_ethnicity, device, epoch):
    model.eval()
    running_loss = 0.0
    age_preds, age_labels = [], []
    gender_preds, gender_labels = [], []
    ethnicity_preds, ethnicity_labels = [], []

    with torch.no_grad():
        for images, age_labels_batch, gender_labels_batch, ethnicity_labels_batch in dataloader:
            images, age_labels_batch, gender_labels_batch, ethnicity_labels_batch = images.to(device), age_labels_batch.to(device), gender_labels_batch.to(device), ethnicity_labels_batch.to(device)

            # Forward pass
            age_pred, gender_pred, ethnicity_pred = model(images)

            # Compute losses
            loss_age = criterion_age(age_pred, age_labels_batch)
            loss_gender = criterion_gender(gender_pred, gender_labels_batch)
            loss_ethnicity = criterion_ethnicity(ethnicity_pred, ethnicity_labels_batch)
            loss = loss_age + loss_gender + loss_ethnicity

            running_loss += loss.item()

            # Store predictions and labels
            age_preds.extend(torch.argmax(age_pred, dim=1).cpu().numpy())
            age_labels.extend(age_labels_batch.cpu().numpy())
            gender_preds.extend(torch.argmax(gender_pred, dim=1).cpu().numpy())
            gender_labels.extend(gender_labels_batch.cpu().numpy())
            ethnicity_preds.extend(torch.argmax(ethnicity_pred, dim=1).cpu().numpy())
            ethnicity_labels.extend(ethnicity_labels_batch.cpu().numpy())

    # Compute metrics
    age_accuracy = accuracy_score(age_labels, age_preds)
    age_precision = precision_score(age_labels, age_preds, average='weighted', zero_division=0)
    age_recall = recall_score(age_labels, age_preds, average='weighted', zero_division=0)
    age_f1 = f1_score(age_labels, age_preds, average='weighted', zero_division=0)

    gender_accuracy = accuracy_score(gender_labels, gender_preds)
    gender_precision = precision_score(gender_labels, gender_preds, average='binary', zero_division=0)
    gender_recall = recall_score(gender_labels, gender_preds, average='binary', zero_division=0)
    gender_f1 = f1_score(gender_labels, gender_preds, average='binary', zero_division=0)

    ethnicity_accuracy = accuracy_score(ethnicity_labels, ethnicity_preds)
    ethnicity_precision = precision_score(ethnicity_labels, ethnicity_preds, average='weighted', zero_division=0)
    ethnicity_recall = recall_score(ethnicity_labels, ethnicity_preds, average='weighted', zero_division=0)
    ethnicity_f1 = f1_score(ethnicity_labels, ethnicity_preds, average='weighted', zero_division=0)

    # Log metrics to TensorBoard
    writer.add_scalar('Validation Loss (Epoch)', running_loss / len(dataloader), epoch)
    writer.add_scalar('Age Accuracy', age_accuracy, epoch)
    writer.add_scalar('Age Precision', age_precision, epoch)
    writer.add_scalar('Age Recall', age_recall, epoch)
    writer.add_scalar('Age F1', age_f1, epoch)
    writer.add_scalar('Gender Accuracy', gender_accuracy, epoch)
    writer.add_scalar('Gender Precision', gender_precision, epoch)
    writer.add_scalar('Gender Recall', gender_recall, epoch)
    writer.add_scalar('Gender F1', gender_f1, epoch)
    writer.add_scalar('Ethnicity Accuracy', ethnicity_accuracy, epoch)
    writer.add_scalar('Ethnicity Precision', ethnicity_precision, epoch)
    writer.add_scalar('Ethnicity Recall', ethnicity_recall, epoch)
    writer.add_scalar('Ethnicity F1', ethnicity_f1, epoch)

    # Return metrics
    metrics = {
        'loss': running_loss / len(dataloader),
        'age_accuracy': age_accuracy,
        'age_precision': age_precision,
        'age_recall': age_recall,
        'age_f1': age_f1,
        'gender_accuracy': gender_accuracy,
        'gender_precision': gender_precision,
        'gender_recall': gender_recall,
        'gender_f1': gender_f1,
        'ethnicity_accuracy': ethnicity_accuracy,
        'ethnicity_precision': ethnicity_precision,
        'ethnicity_recall': ethnicity_recall,
        'ethnicity_f1': ethnicity_f1,
    }
    return metrics

# Training loop with early stopping and learning rate scheduler
best_loss = float('inf')
patience = 5
no_improvement = 0

num_epochs = 50
 
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion_age, criterion_gender, criterion_ethnicity, device, epoch)
    metrics = evaluate(model, test_loader, criterion_age, criterion_gender, criterion_ethnicity, device, epoch)
    
    # Print metrics
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}, Validation Loss: {metrics['loss']:.4f}")
    print("Age Metrics:")
    print(f"  Accuracy: {metrics['age_accuracy']:.4f}, Precision: {metrics['age_precision']:.4f}, Recall: {metrics['age_recall']:.4f}, F1: {metrics['age_f1']:.4f}")
    print("Gender Metrics:")
    print(f"  Accuracy: {metrics['gender_accuracy']:.4f}, Precision: {metrics['gender_precision']:.4f}, Recall: {metrics['gender_recall']:.4f}, F1: {metrics['gender_f1']:.4f}")
    print("Ethnicity Metrics:")
    print(f"  Accuracy: {metrics['ethnicity_accuracy']:.4f}, Precision: {metrics['ethnicity_precision']:.4f}, Recall: {metrics['ethnicity_recall']:.4f}, F1: {metrics['ethnicity_f1']:.4f}")
    print("-" * 50)

    # Early stopping
    if metrics['loss'] < best_loss:
        best_loss = metrics['loss']
        no_improvement = 0
        torch.save(model.module.state_dict(), 'best_model.pth')
    else:
        no_improvement += 1
        if no_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Update learning rate
    # scheduler.step()

# Save the model
torch.save(model.module.state_dict(), 'multitask_mobilenetv2_age_classification.pth')

# Save checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.module.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': metrics['loss'],
}
torch.save(checkpoint, 'checkpoint.pth')

# Close TensorBoard writer
writer.close()