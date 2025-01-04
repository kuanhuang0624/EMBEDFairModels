import pandas as pd
import torch
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import random
from sklearn.model_selection import train_test_split
from dataset import MammoDataset
from tqdm import tqdm
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
train_df = train_df[(train_df['ETHNIC_GROUP_DESC'] == 'Hispanic or Latino') &
                  (train_df['RACE_DESC'] == 'African American  or Black')]
# Additional filtering for test set to include only specific ETHNIC_GROUP_DESC and RACE_DESC
test_df = test_df[(test_df['ETHNIC_GROUP_DESC'] == 'Non-Hispanic or Latino')]
# (test_df['ETHNIC_GROUP_DESC'] == 'Non-Hispanic or Latino')&
# &
# (test_df['RACE_DESC'] == 'Native Hawaiian or Other Pacific Islander')
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
#
# # Modify the final fully connected layer for binary classification
# num_features = model.fc.in_features  # Number of input features to the final fully connected layer
# model.fc = nn.Sequential(
#             nn.Dropout(p=0.4),
#             nn.Linear(num_features, 2)
#         )

model = torch.load("./save_models/swin_256.pt")

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_score, \
    recall_score, confusion_matrix


# Function to calculate metrics
def calculate_metrics(y_true, y_pred, y_prob):
    metrics = {}
    if len(np.unique(y_true)) == 2:
        metrics['AUROC'] = roc_auc_score(y_true, y_prob[:, 1])
        metrics['AUPR'] = average_precision_score(y_true, y_prob[:, 1])
    else:
        metrics['AUROC'] = None
        metrics['AUPR'] = None

    metrics['F1'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['ACC'] = accuracy_score(y_true, y_pred)
    metrics['PRECISION'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['TPR'] = recall_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics['FPR'] = fp / (fp + tn) if (fp + tn) > 0 else None

    return metrics


# Function to evaluate the model on the test set
def evaluate_model(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []
    indices = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader)):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            predicted = np.argmax(probabilities, axis=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted)
            y_prob.extend(probabilities)
            indices.extend(
                dataloader.dataset.dataframe.index[batch_idx * inputs.size(0):(batch_idx + 1) * inputs.size(0)])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    indices = np.array(indices)

    return y_true, y_pred, y_prob, indices


# Load your model
# model = ...  # Load your trained model here

# Evaluate on the whole test set
y_true, y_pred, y_prob, test_indices = evaluate_model(model, test_loader, device)
metrics_all = calculate_metrics(y_true, y_pred, y_prob)
print('Metrics for the whole test set:')
print(metrics_all)
