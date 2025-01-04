import pandas as pd
import torch
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import random
from sklearn.model_selection import train_test_split
from dataset import MammoDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

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

model = torch.load("./save_models/swin_new3.pt")


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

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            predicted = np.argmax(probabilities, axis=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted)
            y_prob.extend(probabilities)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    return y_true, y_pred, y_prob


# Define your subgroups
subgroups = [
    'Hispanic or Latino',
    'Non-Hispanic or Latino',
]

# Dictionary to store EqOdd results
eqodd_results = []

# Loop over each subgroup pair
for i in range(len(subgroups)):
    for j in range(i + 1, len(subgroups)):
        subgroup_a = subgroups[i]
        subgroup_b = subgroups[j]

        # Filter test set for subgroup A
        test_df_a = test_df[
            (test_df['ETHNIC_GROUP_DESC'] == subgroup_a)
            ]

        # Filter test set for subgroup B
        test_df_b = test_df[
            (test_df['ETHNIC_GROUP_DESC'] == subgroup_b)
            ]

        # Create datasets and dataloaders for both subgroups
        test_dataset_a = MammoDataset(test_df_a, False, transform)
        test_dataset_b = MammoDataset(test_df_b, False, transform)

        test_loader_a = data.DataLoader(test_dataset_a, batch_size=32, shuffle=False, num_workers=4)
        test_loader_b = data.DataLoader(test_dataset_b, batch_size=32, shuffle=False, num_workers=4)

        # Evaluate model on both subgroups
        y_true_a, y_pred_a, y_prob_a = evaluate_model(model, test_loader_a, device)
        y_true_b, y_pred_b, y_prob_b = evaluate_model(model, test_loader_b, device)

        # Calculate EqOpp0 and EqOpp1
        fpr_a = np.mean(y_pred_a[y_true_a == 0] == 1)
        fpr_b = np.mean(y_pred_b[y_true_b == 0] == 1)
        eqopp0 = 1 - abs(fpr_a - fpr_b)
        tpr_a = np.mean(y_pred_a[y_true_a == 1] == 1)
        tpr_b = np.mean(y_pred_b[y_true_b == 1] == 1)
        eqopp1 = 1 - abs(tpr_a - tpr_b)
        # eqopp0 = 1 - abs(
        #     np.mean(y_pred_a[y_true_a == 0]) - np.mean(y_pred_b[y_true_b == 0])
        # )
        # eqopp1 = 1 - abs(
        #     np.mean(y_pred_a[y_true_a == 1]) - np.mean(y_pred_b[y_true_b == 1])
        # )

        # Calculate EqOdd
        eqodd = 0.5 * (eqopp0 + eqopp1)

        # Store the results
        eqodd_results.append({
            'Subgroup A': f"{subgroup_a[0]} ({subgroup_a[1]})",
            'Subgroup B': f"{subgroup_b[0]} ({subgroup_b[1]})",
            'EqOpp0': eqopp0,
            'EqOpp1': eqopp1,
            'EqOdd': eqodd
        })

# Convert results to DataFrame
eqodd_df = pd.DataFrame(eqodd_results)

# Display the DataFrame with EqOdd results
print(eqodd_df)
