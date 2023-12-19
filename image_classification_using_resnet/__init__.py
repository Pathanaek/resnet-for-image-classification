# Image Classification Using A Residual Network

# MOUNT TO GOOGLE DRIVE
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
FOLDERNAME = 'acmlab/project'
assert FOLDERNAME is not None, "[!] Enter the foldername."
import sys
PATH = '/content/drive/My Drive/{}'.format(FOLDERNAME)
sys.path.append(PATH)
%cd $PATH


# Math libraries
import numpy as np
import torch
from torch import nn
import utils

# Data processing
import pandas as pd

# Data transformations
import torchvision
from torchvision import transforms
from torchvision.transforms import v2
from torch import Tensor
from torch.utils.data import Dataset

# Loading images
from PIL import Image
import h5py

# Plotting
import matplotlib.pyplot as plt

# Progress bars
import tqdm
import time
from datetime import date


device = torch.device('cuda')

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        my_tuple = utils.load_data()
        self.images = my_tuple[0]
        self.labels = my_tuple[1]
        self.categories = my_tuple[2]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def num_categories(self):
        return len(set(self.categories))

    def get_category(self, label):
        new_list = list(dict.fromkeys(self.labels))
        return self.categories[new_list.index(label)]

    def get_image(self, idx):
        return self.images[idx]

    def get_label(self, idx):
        return self.labels[idx]

    def __getitem__(self, idx):
        my_tuple = (self.transform(self.images[idx]), self.labels[idx])
        return my_tuple

    def display(self, idx):
        display(self.get_image(idx))

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])
all_data = ImageDataset(transform)



class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding='same',
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding='same',
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(p=0.4)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(3, 3),
                    padding='same',
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = self.downsample(x) if self.downsample else x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = self.relu(x + identity)
        return x


class ImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, kernel_size=3, padding=1
        )
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.layer3 = nn.Sequential(
            ResidualBlock(32, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.layer4 = nn.Sequential(
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.layer5 = nn.Sequential(
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.layer6 = nn.Sequential(
            ResidualBlock(256, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.flatten = nn.Flatten(1)
        self.dropout = nn.Dropout(p=0.4)
        self.linear = nn.Linear(
            in_features=int(18432),
            out_features=20
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x


# Hyperparameters / transformations

batch_size = 32
learning_rate = 0.001
num_epochs = 250

aug = torchvision.transforms.Compose([
    torchvision.transforms.v2.RandomHorizontalFlip(),
    torchvision.transforms.v2.RandomRotation(10),
    torchvision.transforms.v2.RandomResizedCrop(size=(224, 224), antialias=True),
    torchvision.transforms.Normalize([0, 0, 0], [1, 1, 1])
])


# Create train and val sets

class CustomDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


train_len = int(0.8 * len(all_data)) # 80% training
val_len = len(all_data) - train_len # 20% validation

generator1 = torch.Generator().manual_seed(42)
train_set, val_set = torch.utils.data.random_split(all_data, [train_len, val_len], generator=generator1)

train_dataset = CustomDataset(train_set, transform=aug)
val_dataset = CustomDataset(val_set, transform=None)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Eval function

def evaluate(model, data_loader, name):
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.float()
            labels = labels.long()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    if name == "val":
      if 89.7 <= accuracy < 90:
          torch.save(model.state_dict(), "final_89.7_resnet.pth")
      if 90 <= accuracy < 91:
          torch.save(model.state_dict(), "final_90_resnet.pth")
      if 91 <= accuracy < 92:
          torch.save(model.state_dict(), "final_91_resnet.pth")
    print(f'Accuracy of the network on the {total} {name} images: {100 * correct / total}%')


# Training loop

model = ImageModel().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 250, eta_min = 1e-7)
start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.float()
        labels = labels.long()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        average_loss = epoch_loss / len(train_loader)
    evaluate(model, train_loader, name="train")
    evaluate(model, val_loader, name="val")
    print(f'Learning rate: { optimizer.param_groups[0]["lr"] }')
    lr_scheduler.step()
    average_loss = epoch_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')

end_time = time.time()
print(f"Total training time: {end_time - start_time} sec")


# Predict function

def load_model():
    model_path = 'PUT_SAVED_WEIGHTS_HERE'
    model = ImageModel()
    model.load_state_dict(torch.load(model_path))
    return model


model = load_model()
model.eval()


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
    ])
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)


def predict(image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()


image_path = 'PUT_IMAGE_PATH_HERE'
prediction = predict(image_path)
print(f'Predicted class: {prediction}')

