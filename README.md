# EMBEDFairModels: Fairness Evaluation in Classification Models

This repository contains code and resources for training and evaluating the fairness of classification models, specifically ResNet and Swin Transformer V2, using the [EMBED Open Data](https://github.com/Emory-HITI/EMBED_Open_Data) dataset. The project focuses on assessing model performance across different racial and ethnic groups to ensure equitable outcomes.

**Our paper titled "Investigating the Fairness of Deep Learning Models in Breast Cancer Diagnosis Based on Race and Ethnicity" has been accepted by the AAAI Fall Symposium Series 2024 Machine Intelligence for Equitable Global Health.**  
[Read our paper here](https://ojs.aaai.org/index.php/AAAI-SS/article/view/31806/33973)

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Models](#models)
- [Fairness Evaluation](#fairness-evaluation)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Introduction
The goal of this project is to train classification models and evaluate their fairness across racial and ethnic groups. We focus on measuring fairness using the Equalized Odds (EqOdd) metric, which compares the true positive and false positive rates across subgroups.

## Dataset
We utilize the [EMBED Open Data](https://github.com/Emory-HITI/EMBED_Open_Data) dataset, which includes medical imaging data categorized by race and ethnicity. This dataset allows for comprehensive analysis of model performance disparities across diverse demographic groups.

## Models
The models trained and evaluated in this project include:
- **ResNet**: A convolutional neural network known for its deep architecture and residual connections.
- **Swin Transformer V2**: A state-of-the-art transformer model adapted for image classification tasks.

## Fairness Evaluation
We assess model fairness using the Equalized Odds (EqOdd) metric for different subgroups:
- **Hispanic or Latino vs. Non-Hispanic or Latino**

Analysis is performed using scripts such as `EqOdd_Hispanic_Non_Hispanic.py` and `eval_race_ethnic.py`.

## Installation
To set up the environment, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/YourUsername/EMBEDFairModels.git
    ```
2. Ensure you have access to the [EMBED Open Data](https://github.com/Emory-HITI/EMBED_Open_Data) dataset.

## Usage
1. **Dataloader**: Load the dataset using `dataset.py`.
2. **Select Images for Training**: Filter images for training using `filter_image.py`.
3. **Convert DICOM to PNG**: Convert DICOM images to PNG format using `DICOM_TO_PNG.py`.
4. **Train Models**:
   - Train ResNet using `train.py`.
   - Train Swin Transformer using `train_swin.py`.
5. **Evaluate Metrics**: Evaluate model metrics using `eval_race_ethnic.py`.
6. **Fairness Evaluation**: Evaluate Equalized Odds using `EqOdd_Hispanic_Non_Hispanic.py`.

## Results
The fairness evaluation showed variations in model performance across different racial and ethnic groups, particularly highlighting areas where the models fail to maintain equitable performance.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for improvements or bug fixes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation
If you use this repository, please cite our paper:
```bibtex
@inproceedings{huang2024investigating,
  title={Investigating the Fairness of Deep Learning Models in Breast Cancer Diagnosis Based on Race and Ethnicity},
  author={Huang, Kuan and Wang, Yingfeng and Xu, Meng},
  booktitle={Proceedings of the AAAI Symposium Series},
  volume={4},
  number={1},
  pages={303--307},
  year={2024}
}
