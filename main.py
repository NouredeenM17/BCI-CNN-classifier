import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import os
from PIL import Image
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime


# Dataset Class

class MedicalDataset(Dataset):
    def __init__(self, root_dirs, img_size=224):
        self.image_paths = []
        self.labels = []
        
        # loading from original train and test directories
        for root_dir in root_dirs:
            for filename in os.listdir(root_dir):
                if filename.endswith('.png'):
                    # Extract label from filename
                    class_str = filename.split('_')[-1].split('.')[0]
                    class_mapping = {'0': 0, '1+': 1, '2+': 2, '3+': 3}
                    label = class_mapping[class_str]
                    
                    self.image_paths.append(os.path.join(root_dir, filename))
                    self.labels.append(label)
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        label = self.labels[idx]
        return image, label


# Data Preparation and Splitting

# Original paths
original_train_dir = 'C:/Users/Techno/Desktop/Deep Learning Project 1/BCI_dataset/train'
original_test_dir = 'C:/Users/Techno/Desktop/Deep Learning Project 1/BCI_dataset/train'

# Create combined dataset
combined_dataset = MedicalDataset([original_train_dir, original_test_dir])

# Stratified split into 70-15-15
indices = np.arange(len(combined_dataset))
train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.3,
    stratify=combined_dataset.labels,
    random_state=42
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.5,
    stratify=np.array(combined_dataset.labels)[temp_idx],
    random_state=42
)

# Create subsets
train_dataset = Subset(combined_dataset, train_idx)
val_dataset = Subset(combined_dataset, val_idx)
test_dataset = Subset(combined_dataset, test_idx)

# Create dataloaders
batch_size = 32
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Calculate Class Weights

# Check for CUDA
print(torch.__version__)
print(f'cuda: {torch.cuda.is_available()}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Check class distribution
train_labels = [combined_dataset.labels[i] for i in train_idx] 
val_labels = [combined_dataset.labels[i] for i in val_idx]
test_labels = [combined_dataset.labels[i] for i in test_idx]

print("Data distribution on Train:\n", Counter(train_labels))
print("Data distribution on Val:\n", Counter(val_labels))
print("Data distribution on Test:\n", Counter(test_labels))

# Calculate class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)


# Model Architecture

class CustomCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, 64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.ReLU(),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = CustomCNN(num_classes=4)

# Training setup

model = model.to(device)
print(model)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# TensorBoard setup
log_dir = f"logs/class_weights_dropout_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir=log_dir)

# Early stopping setup
best_val_loss = float('inf')
best_val_accuracy = float('-inf')
patience = 10
counter = 0


# Training Loop

for epoch in range(200):
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    # Validation
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    
    # Log metrics
    writer.add_scalars('Loss', {
        'train': train_loss/len(train_loader),
        'val': val_loss/len(val_loader)
    }, epoch)
    
    writer.flush()

    writer.add_scalars('Accuracy', {
        'train': correct_train/total_train,
        'val': correct_val/total_val
    }, epoch)
    
    writer.flush()

    print(f'Epoch: {epoch}')
    print(f'Val Loss: {val_loss/len(val_loader)}')
    print(f'Val Acc: {correct_val/total_val}')

    # # Early stopping with accuracy
    # if correct_val/total_val > best_val_accuracy:
    #     best_val_accuracy = correct_val/total_val
    #     torch.save(model.state_dict(), 'best_model.pth')
    #     counter = 0
    # else:
    #     counter += 1
    #     if counter >= patience:
    #         print(f'Early stopping at epoch {epoch}')
    #         break

    # Early stopping with loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break


# Evaluation

# Load best model
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

all_labels = []
all_preds = []
total_loss = 0.0
total_correct = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)  # Move labels to device
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Accumulate loss and accuracy
        total_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        total_correct += (preds == labels).sum().item()
        
        # Store for confusion matrix
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# Calculate final metrics
test_loss = total_loss / len(test_loader.dataset)
test_acc = total_correct / len(test_loader.dataset)

print(f'\nTest Results:')
print(f'Loss: {test_loss:.4f}')
print(f'Accuracy: {test_acc:.4f}')
print(f'Total Samples: {len(test_loader.dataset)}')

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['0', '1+', '2+', '3+'],
            yticklabels=['0', '1+', '2+', '3+'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Close TensorBoard writer
writer.close()