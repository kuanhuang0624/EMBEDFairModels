import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Explicitly set the Matplotlib backend
plt.switch_backend('agg')  # Use 'agg' for non-GUI backend

# Load the CSV files
clinical_file_path = "../tables/EMBED_OpenData_clinical_reduced.csv"
metadata_file_path = "../tables/EMBED_OpenData_metadata_reduced.csv"

df = pd.read_csv(clinical_file_path, dtype={'20': str}, low_memory=False)
metadata_df = pd.read_csv(metadata_file_path)

# Remove rows with same empi_anon, acc_anon, and side but different asses
df_unique = df.drop_duplicates(subset=["empi_anon", "acc_anon", "side"], keep=False)

# Remove rows with specific values in RACE_DESC
# Remove rows with specific values in RACE_DESC and empty path_severity
filtered_df = df_unique[
    ~df_unique["RACE_DESC"].isin(["Unknown, Unavailable or Unreported", "Not Recorded", "Patient Declines"]) &
    df_unique["path_severity"].notna() & (df_unique["path_severity"] != '')
]
# Filter out classes with fewer than 2 members
race_counts = filtered_df["RACE_DESC"].value_counts()
valid_races = race_counts[race_counts >= 2].index
filtered_df = filtered_df[filtered_df["RACE_DESC"].isin(valid_races)]

# Remove rows with 'asses' equal to "X"
filtered_df = filtered_df[filtered_df["asses"] != "X"]


# Filter metadata for images with FinalImageType = "2D"
metadata_df = metadata_df[metadata_df["FinalImageType"] == "2D"]


# Function to merge train or test dataframe with metadata to get image paths and labels
def merge_with_metadata(df, metadata_df):
    # Merge with metadata to get image paths
    merged_df = df.merge(metadata_df, on=["empi_anon", "acc_anon"])

    # Select relevant columns
    result_df = merged_df[["empi_anon", "acc_anon", "anon_dicom_path", "asses", "RACE_DESC", "path_severity"]]
    return result_df

# Merge clinical dataframes with metadata
merged_df = merge_with_metadata(filtered_df, metadata_df)


# Remove duplicate empi_anon entries to count unique RACE_DESC
unique_empi_anon_df = merged_df.drop_duplicates(subset=["empi_anon"])

# Count unique RACE_DESC based on unique empi_anon
race_counts = unique_empi_anon_df["RACE_DESC"].value_counts()

# # Save the merged dataframes for further use
merged_df.to_csv('filtered_images_path_labels.csv', index=False)

# Display the counts
print(f"Filtered set size: {merged_df.shape[0]}")
print(race_counts)


# Count unique 'asses' labels and the number of images
asses_counts = merged_df["asses"].value_counts()

# Display the counts
print("'asses' label counts:")
print(asses_counts)
