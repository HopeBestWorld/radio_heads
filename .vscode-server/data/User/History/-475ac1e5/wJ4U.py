import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

# -------------------------
# Device
# -------------------------
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(32)

# -------------------------
# Paths
# -------------------------
train_excel = 'train/class_label.xlsx'
test_excel = 'test/class_label.csv'
train_dir = 'train'
test_dir = 'test'

# -------------------------
# Dataset
# -------------------------
class ExcelLabelDataset(Dataset):
    def __init__(self, file_path, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        if file_path.endswith('.csv'):
            self.data = pd.read_csv(file_path)
        else:
            self.data = pd.read_excel(file_path, engine='openpyxl')

        self.data.columns = self.data.columns.str.strip()
        print("Columns found:", self.data.columns.tolist())

        label_col = 'Class label (whether polycsytic ovary is visible or not visible)'

        # Drop missing labels and paths
        self.data = self.data.dropna(subset=[label_col, 'imagePath'])
        self.data['imagePath'] = self.data['imagePath'].astype(str)
        self.data = self.data[self.data['imagePath'].str.strip() != '']
        self.data = self.data[self.data['imagePath'].apply(lambda x: os.path.isfile(os.path.join(root_dir, str(x))))]

        # Map labels to 0 and 1
        self.data[label_col] = self.data[label_col].map({'Visible': 1, 'Not-visible': 0})
        self.label_col = label_col

        print(f"Number of samples after cleanup: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, row['imagePath'])
        image = Image.open(img_path).convert('RGB')
        label = int(row[self.label_col])
        if self.transform:
            image = self.transform(image)
        return image, label

# -------------------------
# Transforms
# -------------------------
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -------------------------
# Datasets & Loaders
# -------------------------
train_dataset = ExcelLabelDataset(train_excel, train_dir, transform=train_transforms)
test_dataset = ExcelLabelDataset(test_excel, test_dir, transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# -------------------------
# Model
# -------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += shortcut
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self.make_layer(64, 64, 2)
        self.layer2 = self.make_layer(64, 128, 2, stride=2)
        self.layer3 = self.make_layer(128, 256, 2, stride=2)
        self.layer4 = self.make_layer(256, 512, 2, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 2)

    def make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = ResNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------
# Evaluation Metrics
# -------------------------
def evaluate_model(model, val_loader):
    model.eval()
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = nn.functional.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    acc = (y_true == y_pred).mean()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.0

    return acc, precision, recall, f1, auc, y_true, y_pred

# -------------------------
# Training Loop with Early Stopping
# -------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=5):
    best_val_f1 = 0.0
    best_model_path = 'best_model.pkl'
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total

        val_acc, val_prec, val_recall, val_f1, val_auc, _, _ = evaluate_model(model, val_loader)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, '
              f'Precision: {val_prec:.4f}, Recall: {val_recall:.4f}, '
              f'F1: {val_f1:.4f}, AUC: {val_auc:.4f}')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Best model saved with Val F1: {best_val_f1:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"⏹ Early stopping triggered after {patience} epochs without improvement.")
                break

# -------------------------
# Train the model
# -------------------------
train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=50, patience=5)

# -------------------------
# Load best model & evaluate
# -------------------------
model.load_state_dict(torch.load('best_model.pkl'))
model.eval()
val_acc, val_prec, val_recall, val_f1, val_auc, y_true, y_pred = evaluate_model(model, test_loader)

print("\n✅ Final Test Metrics:")
print(f"Accuracy: {val_acc:.4f}")
print(f"Precision: {val_prec:.4f}")
print(f"Recall: {val_recall:.4f}")
print(f"F1-score: {val_f1:.4f}")
print(f"AUC: {val_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["not visible", "visible"]))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=["not visible", "visible"], yticklabels=["not visible", "visible"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
