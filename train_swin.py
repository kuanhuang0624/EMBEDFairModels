import pandas as pd
import torch
import numpy as np
import torch.utils.data as data
from torchvision import transforms
from tqdm import tqdm
import random
import torchvision.models as models
import torch.optim as optim
from sklearn.model_selection import train_test_split
from dataset import MammoDataset
import torch.nn as nn

is_cuda_available = torch.cuda.is_available()

device = torch.device("cuda" if is_cuda_available else "cpu")

# set up random seed

seed_value = 42

torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)  # if using multiple GPUs
np.random.seed(seed_value)
random.seed(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load the filtered CSV file
file_path = "unique_images_filtered.csv"
df = pd.read_csv(file_path)

# Update paths
df['anon_dicom_path'] = df['anon_dicom_path'].str.replace('/mnt/NAS2/mammo/anon_dicom/', './png/')
df['anon_dicom_path'] = df['anon_dicom_path'].str.replace('.dcm', '.png')

# Select rows with path_severity = 0, 1, 4
df = df[df['path_severity'].isin([0, 1, 4])]

# Set the label column
df['label'] = df['path_severity'].apply(lambda x: 0 if x in [0, 1] else 1)

# # Create labels based on path_severity
# df['label'] = df['path_severity'].apply(lambda x: 1 if x in [0, 1] else 0)
df = df.dropna(subset=['RACE_DESC'])

# Split the dataset ensuring the balance of RACE_DESC
train_df, test_df = train_test_split(
    df,
    test_size=0.3,
    random_state=42,
    stratify=df['RACE_DESC']
)

transform = transforms.Compose([
    transforms.ToTensor()
])

# Create datasets
train_dataset = MammoDataset(train_df, True, transform)
test_dataset = MammoDataset(test_df, False, transform)

# Create dataloaders
train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Print the number of samples in training set and testing set
print('Training samples #: ', len(train_dataset))
print('Test samples #: ', len(test_dataset))


# model = models.resnet50(pretrained=True)
import timm
model = timm.create_model('swin_tiny_patch4_window7_224.ms_in1k', pretrained=True, num_classes=2)

# Modify the final fully connected layer for binary classification
# num_features = model.fc.in_features  # Number of input features to the final fully connected layer
# model.fc = nn.Sequential(
#             nn.Dropout(p=0.4),
#             nn.Linear(num_features, 2)
#         )

model = model.to(device)
# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-2)

# loss
loss_function = torch.nn.CrossEntropyLoss()


# Train the model
epoch_training_loss = []
epoch_test_loss = []
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
def calculate_accuracy(output, labels):
    _, preds = torch.max(output, 1)
    return (preds == labels).sum().item() / labels.size(0)


for epoch in range(2):
    model.train()
    train_loss = []
    train_correct = 0
    train_total = 0

    # Training loop with tqdm
    for batch_idx, (image, label) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()

        # Forward pass
        output = model(image)

        # Calculate loss
        loss = loss_function(output, label)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

        # Calculate training accuracy
        train_correct += calculate_accuracy(output, label) * label.size(0)
        train_total += label.size(0)

        # Update the progress bar with the current training accuracy
        train_accuracy = train_correct / train_total
        tqdm.write(f'Batch {batch_idx+1}, Train loss: {loss.item():.4f}, Train accuracy: {train_accuracy:.4f}')

    train_accuracy = train_correct / train_total
    print(f'Epoch {epoch+1}, Train loss: {np.mean(train_loss):.4f}, Train accuracy: {train_accuracy:.4f}')
    epoch_training_loss.append(np.mean(train_loss))
    scheduler.step()

    # Evaluate on test data
    model.eval()
    test_loss = []
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(tqdm(test_loader, desc=f"Testing Epoch {epoch+1}")):
            image, label = image.to(device), label.to(device)
            y_predict = model(image)
            loss = loss_function(y_predict, label)
            test_loss.append(loss.item())

            # Calculate test accuracy
            test_correct += calculate_accuracy(y_predict, label) * label.size(0)
            test_total += label.size(0)

            # Update the progress bar with the current test accuracy
            test_accuracy = test_correct / test_total
            tqdm.write(f'Batch {batch_idx+1}, Test loss: {loss.item():.4f}, Test accuracy: {test_accuracy:.4f}')

    test_accuracy = test_correct / test_total
    print(f'Epoch {epoch+1}, Test loss: {np.mean(test_loss):.4f}, Test accuracy: {test_accuracy:.4f}')
    epoch_test_loss.append(np.mean(test_loss))
# Draw curves here

import matplotlib.pyplot as plt

plt.figure(0)
plt.plot(epoch_training_loss)
plt.plot(epoch_test_loss)
plt.show()
torch.save(model, "./save_models/swin_256.pt")