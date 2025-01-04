import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import random
import cv2

# Define custom dataset
class MammoDataset(Dataset):
    def __init__(self, dataframe, is_augment, transform=None, img_width=224, img_height=224):
        self.dataframe = dataframe
        self.transform = transform
        self.is_augment = is_augment
        self.img_width = img_width
        self.img_height = img_height

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = row['anon_dicom_path']
        label = row['label']


        image = Image.open(image_path).convert("RGB").resize((self.img_height, self.img_width))
        image = np.array(image)
        img_new = np.float32(image)
        img_new = img_new / 255

        if self.is_augment == True:
            flipCode = random.choice([-1, 0, 1, 2, 3])
            if flipCode == 2:
                height, width = self.img_height, self.img_width
                center = (width / 2, height / 2)
                degree = random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]) * 2
                M = cv2.getRotationMatrix2D(center, degree, 1.0)
                img_new = cv2.warpAffine(img_new, M, (height, width))
            elif flipCode != 3:
                img_new = cv2.flip(img_new, flipCode)

        if self.transform:
            img_new = self.transform(img_new)

        return img_new, label


if __name__ == "__main__":
    # Load the filtered CSV file
    file_path = "filtered_images_path_labels.csv"
    df = pd.read_csv(file_path)

    # Update paths
    df['anon_dicom_path'] = df['anon_dicom_path'].str.replace('/mnt/NAS2/mammo/anon_dicom/', './png/')
    df['anon_dicom_path'] = df['anon_dicom_path'].str.replace('.dcm', '.png')
    # Create labels based on path_severity
    df['label'] = df['path_severity'].apply(lambda x: 1 if x in [0, 1] else 0)

    # Split the dataset ensuring the balance of RACE_DESC
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
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
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Example of how to use the dataloaders
    for images, labels in train_loader:
        print(images.shape, labels.shape)
        image = np.uint8((images[0][0].numpy() + 1) * 127.5)
        plt.figure(0)
        plt.imshow(image, cmap="gray")
        plt.show()

