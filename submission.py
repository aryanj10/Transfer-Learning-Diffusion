import os
import time
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

#################################
# GLOBAL SETUP
#################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

#################################
# PART 1: Dataset and Model Setup
#################################
data_dir = './data/hymenoptera_data'

# Transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

print("[INFO] Loading datasets...")
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ['train', 'val']
}

dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    for x in ['train', 'val']
}

class_names = image_datasets['train'].classes
num_classes = len(class_names)

#################################
# PART 1: Training Function
#################################
def train_model(model, dataloaders, criterion, optimizer, scheduler=None, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}\n{"-"*20}')

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f"[{phase.upper()}]"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train' and scheduler:
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'\n✅ Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f'🏆 Best Val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model

#################################
# PART 1: Finetune All Layers
#################################
print("\n[INFO] Finetuning all layers...")
model_ft = models.resnet18(pretrained=True)
model_ft.fc = nn.Linear(model_ft.fc.in_features, num_classes)
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()
optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
scheduler_ft = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, dataloaders, criterion, optimizer_ft, scheduler_ft, num_epochs=25)

#################################
# PART 1: Feature Extractor Mode
#################################
print("\n[INFO] Training feature extractor (freeze all except FC)...")
model_conv = models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False
model_conv.fc = nn.Linear(model_conv.fc.in_features, num_classes)
model_conv = model_conv.to(device)

optimizer_conv = torch.optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
scheduler_conv = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_conv = train_model(model_conv, dataloaders, criterion, optimizer_conv, scheduler_conv, num_epochs=25)

#################################
# PART 1: Evaluation
#################################
def evaluate_model(model, dataloader, class_names, title="Model Evaluation"):
    print(f"\n[INFO] Evaluating model: {title}")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n📄 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix: {title}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

evaluate_model(model_ft, dataloaders['val'], class_names, title="Fine-tuned ResNet18")
evaluate_model(model_conv, dataloaders['val'], class_names, title="Feature Extractor ResNet18")



#################################
# PART 2: Diffusion Sampling
#################################

from diffusers import DDIMPipeline
import torch
from PIL import Image
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Set save path
save_path = './data/cats_real_fake/fake_cats'
os.makedirs(save_path, exist_ok=True)

# Load pipeline 
# Load the pipeline (NO use_safetensors)

pipe = DDIMPipeline.from_pretrained("google/ddpm-cat-256").to(device)
pipe.to(device)

# Generate 150 fake images
for i in range(150):
    image = pipe(num_inference_steps=25).images[0]
    image.save(os.path.join(save_path, f"fake_cat_{i}.png"))
    print(f"✅ Saved fake_cat_{i}.png")
print("✅ All images saved successfully.")



#################################
# PART 2: Moving 100 fake images for training and 50 for validation
#################################

import os
import shutil

# Paths
fake_src = './data/cats_real_fake/fake_cats'
fake_train_dst = './data/cats_real_fake/fake_cats/train/fake'
fake_val_dst = './data/cats_real_fake/fake_cats/val/fake'

# Create train/val folders
os.makedirs(fake_train_dst, exist_ok=True)
os.makedirs(fake_val_dst, exist_ok=True)

# Get all image files sorted by name
all_fake_images = sorted([
    f for f in os.listdir(fake_src)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])

# Move first 100 to train folder
for f in all_fake_images[:100]:
    shutil.move(os.path.join(fake_src, f), os.path.join(fake_train_dst, f))

# Move next 50 (not from the first 100) to val folder
for f in all_fake_images[100:150]:
    shutil.move(os.path.join(fake_src, f), os.path.join(fake_val_dst, f))

print("✅ Moved 100 fake images to train and next 50 to validation folders.")



import os

# Paths
fake_src = './data/cats_real_fake/fake_cats'
fake_train_dst = './data/cats_real_fake/fake_cats/train/fake'
fake_val_dst = './data/cats_real_fake/fake_cats/val/fake'

real_src = './data/cats_real_fake/real_cats'
real_train_dst = './data/cats_real_fake/real_cats/train/real'
real_val_dst = './data/cats_real_fake/real_cats/val/real'

def count_images(folder):
    return len([
        f for f in os.listdir(folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

print("📊 Image Counts:")
print(f"🔹 Source folder     : {count_images(fake_src)} images")
print(f"🔹 Train/fake folder : {count_images(fake_train_dst)} images")
print(f"🔹 Val/fake folder   : {count_images(fake_val_dst)} images")


print("📊 Image Counts:")
print(f"🔹 Source folder     : {count_images(real_src)} images")
print(f"🔹 Train/real folder : {count_images(real_train_dst)} images")
print(f"🔹 Val/real folder   : {count_images(real_val_dst)} images")



import os
import shutil

# Source folders
fake_train_src = './data/cats_real_fake/fake_cats/train/fake'
fake_val_src = './data/cats_real_fake/fake_cats/val/fake'
real_train_src = './data/cats_real_fake/real_cats/train/real'
real_val_src = './data/cats_real_fake/real_cats/val/real'

# Target folders for training and validation (unified structure)
train_fake_dst = './data/cats_real_fake/train/fake'
train_real_dst = './data/cats_real_fake/train/real'
val_fake_dst = './data/cats_real_fake/val/fake'
val_real_dst = './data/cats_real_fake/val/real'

# Make target folders
os.makedirs(train_fake_dst, exist_ok=True)
os.makedirs(train_real_dst, exist_ok=True)
os.makedirs(val_fake_dst, exist_ok=True)
os.makedirs(val_real_dst, exist_ok=True)

# Helper to move files
def move_all(src, dst):
    for f in os.listdir(src):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            shutil.move(os.path.join(src, f), os.path.join(dst, f))

# Move images
move_all(fake_train_src, train_fake_dst)
move_all(fake_val_src, val_fake_dst)
move_all(real_train_src, train_real_dst)
move_all(real_val_src, val_real_dst)

print("✅ All images moved to unified train/val structure for ImageFolder.")



#################################
# PART 2: Fake-vs-Real Classifier
#################################

# Data transforms for classification
cat_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # since CIFAR-style images are less saturated
])

# Load dataset
data_dir = './data/cats_real_fake'

train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=cat_transforms)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=cat_transforms)

# Combine and split into train/val
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=4)


# Define model
model_cat = models.resnet18(pretrained=True)
model_cat.fc = nn.Linear(model_cat.fc.in_features, 2)
model_cat = model_cat.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer_cat = torch.optim.Adam(model_cat.parameters(), lr=1e-4)

# Train loop
from tqdm import tqdm

def train_fake_vs_real(model, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)

        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer_cat.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_cat.step()
            total_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        print(f"📘 Epoch {epoch+1}: Avg Train Loss = {total_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        correct = 0
        total = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]  ", leave=False)
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        print(f"✅ Epoch {epoch+1}: Validation Accuracy = {correct/total:.4f}")

# Optional: Faster DataLoader config


# Run training
train_fake_vs_real(model_cat, train_loader, val_loader)



#################################
# PART 2: Evaluation Report
#################################

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, val_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['real', 'fake']))

    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred: Real', 'Pred: Fake'],
                yticklabels=['True: Real', 'True: Fake'])
    plt.title("Confusion Matrix")
    plt.show()

# Call the evaluator
evaluate_model(model_cat, val_loader)
