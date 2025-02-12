# Test
import pandas as pd

# Load the CSV files
csv_file_path = 'filtered_images_path_labels.csv'

df = pd.read_csv(csv_file_path)

# Define the conditions for filtering
condition1 = (df['path_severity'] == 4) & (df['RACE_DESC'].isin([
    'African American or Black', 'Caucasian or White', 'Asian', 'Native Hawaiian or Other Pacific Islander'
]))

condition2 = df['path_severity'].isin([0, 1])

# Select the rows based on the conditions
filtered_df = df[condition1 | condition2]

# Show the unique RACE_DESC values in the filtered DataFrame
unique_race_desc = filtered_df['RACE_DESC'].unique()
print("Unique RACE_DESC values in the filtered DataFrame:")
print(unique_race_desc)

# Save the filtered DataFrame to a new CSV file
filtered_df.to_csv('final_image_with_label.csv', index=False)

# Display a message indicating that the file has been saved
print(f"Filtered data has been saved to 'final_image_with_label.csv'.")
