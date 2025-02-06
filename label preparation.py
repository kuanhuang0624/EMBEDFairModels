import pandas as pd

# Load the CSV files
train_file_path = 'train_images_path_labels.csv'
test_file_path = 'test_images_path_labels.csv'

train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)

# Concatenate train and test DataFrames
combined_df = pd.concat([train_df, test_df])

# Define the conditions for filtering
condition1 = (combined_df['path_severity'] == 4) & (combined_df['RACE_DESC'].isin([
    'African American or Black', 'Caucasian or White', 'Asian', 'Native Hawaiian or Other Pacific Islander'
]))

condition2 = combined_df['path_severity'].isin([0, 1])

# Select the rows based on the conditions
filtered_df = combined_df[condition1 | condition2]

# Show the unique RACE_DESC values in the filtered DataFrame
unique_race_desc = filtered_df['RACE_DESC'].unique()
print("Unique RACE_DESC values in the filtered DataFrame:")
print(unique_race_desc)

# Save the filtered DataFrame to a new CSV file
filtered_df.to_csv('filtered_images_path_labels.csv', index=False)

# Display a message indicating that the file has been saved
print(f"Filtered data has been saved to 'filtered_images_path_labels.csv'.")
