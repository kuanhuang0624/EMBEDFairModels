import pandas as pd

# Load the CSV files into DataFrames
clinical_file_path = '../tables/EMBED_OpenData_clinical_reduced.csv'
metadata_file_path = '../tables/EMBED_OpenData_metadata_reduced.csv'

df_clinical = pd.read_csv(clinical_file_path)
df_metadata = pd.read_csv(metadata_file_path)

# Filter clinical data
filtered_clinical_df = df_clinical.drop_duplicates(subset=['empi_anon', 'acc_anon', 'side'], keep=False)

# Filter out unwanted RACE_DESC values
unwanted_races = ['Not Recorded', 'Patient Declines', 'Unknown, Unavailable or Unreported', 'Multiple']
filtered_clinical_df = filtered_clinical_df[~filtered_clinical_df['RACE_DESC'].isin(unwanted_races)]

# Filter clinical data where 'path_severity' has a value (not null)
filtered_clinical_df = filtered_clinical_df[filtered_clinical_df['path_severity'].notnull() & (filtered_clinical_df["path_severity"] != '')]

# Filter clinical data to include only specific ETHNIC_GROUP_DESC values
desired_ethnic_groups = ['Hispanic or Latino', 'Non-Hispanic or Latino']
filtered_clinical_df = filtered_clinical_df[filtered_clinical_df['ETHNIC_GROUP_DESC'].isin(desired_ethnic_groups)]

# Filter metadata to include only images with 'FinalImageType == 2D'
filtered_metadata_df = df_metadata[df_metadata['FinalImageType'] == '2D']

# Merge the filtered clinical data with the filtered metadata on 'empi_anon' and 'acc_anon'
merged_df = pd.merge(filtered_metadata_df, filtered_clinical_df, on=['empi_anon', 'acc_anon'], how='inner')

# Remove duplicates to ensure each image is unique
unique_images_df = merged_df.drop_duplicates(subset=['anon_dicom_path'])

# Save the unique images DataFrame to a CSV file
filtered_csv_path = 'unique_images_filtered.csv'
unique_images_df.to_csv(filtered_csv_path, index=False)

print("Filtered data saved to 'unique_images_filtered.csv'")
