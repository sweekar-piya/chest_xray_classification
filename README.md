# Chest X-Ray classification

## Introduction

This is a minor _streamlit_ based personal application project for classification of X-Ray images (modelled in _Pytorch_) into 2 classes:
1. Pneumonia - +ve case (1)
2. Normal - -ve case (0)

Note: The dataset is unbalanced, in sense that there are more +ve samples than -ve ones. This is opposite of real world case. But I have used the following techniques to balance the training:

1. Weighted Binary Cross Entropy Loss
2. Increasing sample through data resampling/transformations.

## Project Structure

```bash
.
└── chest_xray_classification
    ├── asset
    │   └── chest_xray_demo_app.gif 
    ├── data (Keep your data here)
    │   └── 
    ├── models (Save models here)
    │   └──
    ├── notebooks
    │   ├── 2023_10_20_chest_xray_classification_1st_attempt.ipynb (1st attempt)
    │   ├── 2023_10_20_chest_xray_classification-base_cnn.ipynb (Basic CNN model)
    │   ├── 2023_10_20_chest_xray_tf.ipynb (Tried Tensorflow; doesnt work)
    │   └── 2023_10_20_chest_xray_improvements.ipynb (Transfer learning)
    ├── plots
    │   ├── 1st_attempt
    │   ├── baseline_cnn
    │   └── resnet_transfer
    ├── src
    │   ├── models (keep model architecture here)
    │   │   ├── baseline_cnn.py
    │   │   ├── first_model.py
    │   │   └── resnet_transfer.py
    │   └── utils (utility scripts for webapp)
    │       ├── constants.py
    │       └── helper.py
    ├── .gitignore
    ├── requirements.txt
    ├── app.py
    └── README.md
```

## Results from my experimentation

Weighted Accuracy of 3 models I experimented with.

1. First attempt Model - Training __100%__; Testing 62% - Overfitted
2. Basic CNN - Training 94%; Testing 83% - Can be better
3. Transfer Learning (ResNet18) - Training 89.4%, Testing __87%__ - Best one

## How to Run

1. Ensure you have `CUDA 12.1` installated in your workstation.
Create your virtual env `python -m venv .venv`, activate it using `source .venv/bin/activate` or `source .\.venv\Scripts\activate`.
2. Install packages from using `pip install -r requirements.txt`.
3. Download the dataset from https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/ and unzip it under `./data/`.
4. Since, I haven't uploaded the models, you can train your model and place save it under `./models/`
5. Copy the model architecture and create new script under `src/model`, also create a function that instantiates the model object.
6. Import that function to `src/utils/helper.py` and place it in if-else block under `load_model` function.
7. Run `streamlit run app.py`
8. Test the application.

## Demo

<center><img src="./asset/chest_x_ray_app.gif" alt="drawing" width="700"></center>


## Data Acknowledgement

### Content

The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).

Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care.

For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.

### Acknowledgements
Data: https://data.mendeley.com/datasets/rscbjbr9sj/2

License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Citation: http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5

## Improvements To Conduct (If I return to this project):
1. Change images fed in the DL models from RGB to Grayscale for training and prediction.
2. Unify model's output: Currently my vanilla models output sigmoid activated values, while fine-tuned one outputs prediction probability of the image being +ve. Change them reduces code clutter.
3. Finally, I need to introduce dynamic code such that any one can add new models to this project.
4. Create Dockerfile for running anywhere.