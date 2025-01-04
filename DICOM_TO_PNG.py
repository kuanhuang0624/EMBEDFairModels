import os
import pandas as pd
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import cv2

# Function to create directories if they do not exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# Function to convert DICOM to PNG
def dcm_to_png(dcm_path, png_path):
    dcm = pydicom.dcmread(dcm_path)
    image = dcm.pixel_array
    min_val = image.min()
    max_val = image.max()
    image = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    flip_horz, flip_vert = check_dcm(dcm)
    if flip_horz:
        image = np.fliplr(image).copy()
    if flip_vert:
        image = np.flipud(image).copy()

    create_directory(os.path.dirname(png_path))
    cv2.imwrite(png_path, image)

# Function to check DICOM tags and decide on flipping
def check_dcm(imgdcm):
    tags = DCM_Tags(imgdcm)
    if not pd.isnull(tags.orientation):
        if tags.view == 'CC':
            flipHorz = tags.orientation[0] == 'P'
            flipVert = ((tags.laterality == 'L') & (tags.orientation[1] == 'L')) or (
                    (tags.laterality == 'R') & (tags.orientation[1] == 'R'))
        elif tags.view in ['MLO', 'ML']:
            flipHorz = tags.orientation[0] == 'P'
            flipVert = ((tags.laterality == 'L') & (tags.orientation[1] in ['H', 'HL'])) or (
                    (tags.laterality == 'R') & (tags.orientation[1] in ['H', 'HR']))
        else:
            flipHorz, flipVert = False, False
    else:
        flipHorz = (tags.laterality == 'R') & (tags.view in ['CC', 'ML', 'MLO'])
        flipVert = False

    return flipHorz, flipVert


class DCM_Tags:
    def __init__(self, img_dcm):
        try:
            self.laterality = img_dcm.ImageLaterality
        except AttributeError:
            self.laterality = np.nan

        try:
            self.view = img_dcm.ViewPosition
        except AttributeError:
            self.view = np.nan

        try:
            self.orientation = img_dcm.PatientOrientation
        except AttributeError:
            self.orientation = np.nan


# Load the CSV file
file_path = "unique_images_filtered.csv"
df = pd.read_csv(file_path)

# Process each DICOM file and save as PNG
for index, row in df.iterrows():
    dicom_path = row['anon_dicom_path']
    dicom_path = dicom_path.replace('/mnt/NAS2/mammo/anon_dicom/', '../images/')
    # Assuming the column name for DICOM paths in the CSV is 'dicom_path'
    png_path = dicom_path.replace('../images/', './png/').replace('.dcm', '.png')

    try:
        dcm_to_png(dicom_path, png_path)
        print(f'Successfully converted {dicom_path} to {png_path}')
    except Exception as e:
        print(f'Failed to convert {dicom_path}: {e}')
