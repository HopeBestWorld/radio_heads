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
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, precision_recall_curve, auc)

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

print(f"Number of training images: {len(train_dataset)}")
print(f"Number of testing images: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# -------------------------
# Visualization functions
# -------------------------
def plot_class_distribution(dataset, title="Class Distribution", filename="class_distribution.png"):
    labels = [dataset[i][1] for i in range(len(dataset))]
    plt.figure(figsize=(6, 4))
    sns.countplot(x=labels)
    plt.xticks([0, 1], ['Not Visible', 'Visible'])
    plt.title(title)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def show_samples(dataset, num_samples=4, filename="sample_images.png"):
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    for ax, idx in zip(axes, indices):
        img, label = dataset[idx]
        img_np = img.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)
        ax.imshow(img_np)
        ax.set_title("Visible" if label == 1 else "Not Visible")
        ax.axis('off')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# Visualizations
plot_class_distribution(train_dataset, title="Training Set Class Distribution", filename="train_class_distribution.png")
plot_class_distribution(test_dataset, title="Testing Set Class Distribution", filename="test_class_distribution.png")
show_samples(train_dataset, num_samples=6, filename="train_sample_images.png")

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
    """
    Evaluates the model and returns predictions, probabilities, and PR-AUC.
    """
    model.eval()
    y_true, y_pred, y_probs_list = [], [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = nn.functional.softmax(outputs, dim=1)  # keep all class probabilities
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_probs_list.append(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.vstack(y_probs_list)  # shape [n_samples, n_classes]

    # Basic classification metrics
    acc = (y_true == y_pred).mean()
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    # PR-AUC (binary, for positive class "Visible")
    if y_probs.ndim == 2 and y_probs.shape[1] == 2:
        pos_prob = y_probs[:, 1]  # probability of class 1
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, pos_prob)
        pr_auc_score = auc(recall_curve, precision_curve)
    else:
        pr_auc_score = 0.0

    return (acc, precision, recall, f1, pr_auc_score, 
            y_true, y_pred, y_probs, 
            per_class_precision, per_class_recall, per_class_f1)

# -------------------------
# Training Loop with Early Stopping
# -------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=5):
    best_val_f1 = 0.0
    best_model_path = 'best_model.pkl'
    epochs_no_improve = 0

    # Lists to store metrics per epoch
    train_acc_list = []
    val_acc_list = []

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
        train_acc_list.append(train_acc)

        val_acc, val_prec, val_recall, val_f1, val_auc, _, _, _, _, _, _ = evaluate_model(model, val_loader)
        val_acc_list.append(val_acc)

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

    # Plot accuracy curve
    plt.figure(figsize=(8, 6))
    plt.plot(train_acc_list, label='Train Accuracy')
    plt.plot(val_acc_list, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_curve.png", dpi=300, bbox_inches='tight')
    plt.show()

# -------------------------
# Train the model
# -------------------------
train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=50, patience=5)

# -------------------------
# Load best model & evaluate
# -------------------------
model.load_state_dict(torch.load('best_model.pkl'))
model.eval()
val_acc, val_prec, val_recall, val_f1, val_auc, y_true, y_pred, y_probs, per_class_precision, per_class_recall, per_class_f1 = evaluate_model(model, test_loader)

print("\n✅ Final Test Metrics:")
print(f"Accuracy: {val_acc:.4f}")
print(f"Precision: {val_prec:.4f}")
print(f"Recall: {val_recall:.4f}")
print(f"F1-score: {val_f1:.4f}")
print(f"AUC: {val_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["not visible", "visible"]))

# -------------------------
# Confusion Matrix
# -------------------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=["not visible", "visible"], yticklabels=["not visible", "visible"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()

# -------------------------
# Grouped bar chart for precision, recall, F1
# -------------------------
def plot_metrics(per_class_precision, per_class_recall, per_class_f1, class_names=["Not Visible", "Visible"], filename="metrics_per_class.png"):
    x = np.arange(len(class_names))
    width = 0.25

    plt.figure(figsize=(8, 6))
    plt.bar(x - width, per_class_precision, width, label='Precision')
    plt.bar(x, per_class_recall, width, label='Recall')
    plt.bar(x + width, per_class_f1, width, label='F1-score')

    plt.xticks(x, class_names)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Precision, Recall, F1 per Class")
    plt.legend()
    for i in range(len(class_names)):
        plt.text(i - width, per_class_precision[i]+0.02, f"{per_class_precision[i]:.2f}", ha='center')
        plt.text(i, per_class_recall[i]+0.02, f"{per_class_recall[i]:.2f}", ha='center')
        plt.text(i + width, per_class_f1[i]+0.02, f"{per_class_f1[i]:.2f}", ha='center')

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

plot_metrics(per_class_precision, per_class_recall, per_class_f1, class_names=["Not Visible", "Visible"])

# -------------------------
# Precision-Recall Curve
# -------------------------
def plot_precision_recall(y_true, y_probs, class_names=None, filename="precision_recall_curve.png"):
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    if y_probs.ndim == 2 and y_probs.shape[1] == 2:
        # Binary classification: take the probability of class 1 (positive)
        pos_prob = y_probs[:, 1]
    else:
        pos_prob = y_probs  # already shape [n_samples]

    if class_names is None:
        class_names = ["Not Visible", "Visible"]

    precision, recall, _ = precision_recall_curve(y_true, pos_prob)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, lw=2, label=f'{class_names[1]} (AUC={pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

plot_precision_recall(y_true, y_probs, class_names=["Not Visible", "Visible"])

# -------------------------
# Grad-CAM
# -------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(input_tensor.to(device))
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        loss = output[0, target_class]
        loss.backward()

        # Compute weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = nn.functional.relu(cam)
        cam = nn.functional.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # normalize
        return cam

def show_gradcam(img, cam, title="Grad-CAM", filename="gradcam.png"):
    img_np = img.permute(1, 2, 0).numpy()
    img_np = np.clip(img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)

    plt.figure(figsize=(6,6))
    plt.imshow(img_np)
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.title(title)
    plt.axis('off')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# -------------------------
# Example Usage
# -------------------------
# Pick an image from the test dataset
sample_img, sample_label = test_dataset[0]
input_tensor = sample_img.unsqueeze(0).to(device)

# Initialize Grad-CAM with the last convolutional layer
grad_cam = GradCAM(model, model.layer4)

# Generate CAM for the positive class (Visible)
cam = grad_cam.generate_cam(input_tensor, target_class=1)

# Display Grad-CAM overlay
show_gradcam(sample_img, cam, title=f"Grad-CAM - True Label: {'Visible' if sample_label==1 else 'Not Visible'}")

# -------------------------
# Generate Grad-CAM for all False Negatives
# -------------------------
def gradcam_false_negatives(model, dataset, y_true, y_pred, target_layer, save_dir="gradcam_false_negatives"):
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    grad_cam = GradCAM(model, target_layer)
    
    for idx, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
        if true_label == 1 and pred_label == 0:  # False Negative
            img, _ = dataset[idx]
            input_tensor = img.unsqueeze(0).to(device)
            cam = grad_cam.generate_cam(input_tensor, target_class=1)  # positive class
            
            filename = os.path.join(save_dir, f"fn_{idx}.png")
            show_gradcam(img, cam, title=f"Grad-CAM False Negative #{idx}", filename=filename)
            print(f"Saved Grad-CAM for False Negative #{idx} -> {filename}")

# Example usage after evaluation
gradcam_false_negatives(model, test_dataset, y_true, y_pred, target_layer=model.layer4)
