# Beyond Black-Box: A Random Forest and SHAP Cardiovascular System for Personalized Lifestyle Interventions

## Overview
This repository contains the computational artifacts, datasets, and source code for the research article titled **"Beyond Black-Box: A Random Forest and SHAP Cardiovascular System for Personalized Lifestyle Interventions."** This project introduces a hybrid cardiovascular risk prediction architecture that integrates the analytical classification accuracy of a Random Forest ensemble, the transparency of Explainable AI (SHAP - SHapley Additive exPlanations), and a rule-based expert system. The system is designed not only to predict cardiovascular risk but also to empower patients by generating transparent, actionable, and personalized self-managed lifestyle interventions.

## Repository Structure
The project directory is systematically organized as follows to ensure seamless local execution and deployment:

```text
CardioCare-Expert-System/
│
├── dataset/
│   └── cardio_cleaned.csv         # The preprocessed corpus including the engineered BMI feature
│
├── notebooks/
│   └── cardio_training.ipynb      # Source code for data cleaning, EDA, and model training
│
├── app.py                         # Main Streamlit web application and Rule-Based Engine
├── model_kardio_rf.zip            # The compressed Random Forest predictive model (Root directory)
├── requirements.txt               # Required Python dependencies
└── README.md                      # Project documentation
```
Prerequisites
To run this application locally, ensure that you have Python 3.8 or higher installed on your system.

Installation and Execution Guide
Follow these sequential steps to deploy the application on your local machine:

1. Clone the Repository
   git clone [https://github.com/xbracaa/CardioCare-Expert-System.git](https://github.com/xbracaa/CardioCare-Expert-System.git)
cd CardioCare-Expert-System
2. Extract the Predictive Model
Due to GitHub's file size limitations, the serialized machine learning model has been compressed into a .zip archive.

Extract the model_kardio_rf.zip file directly within the root directory.

Ensure that the extracted model_kardio_rf.pkl file is placed in the same folder level as app.py.

3. Install Dependencies
Install the required libraries as specified in the requirements file:

Bash
pip install -r requirements.txt
4. Run the Application
Execute the main script using Streamlit:

Bash
streamlit run app.py
The application will automatically launch in your default web browser.

Dataset
The original computational baseline dataset is publicly accessible via Kaggle (Sulianova Cardiovascular Disease Dataset). The refined dataset (cardio_cleaned.csv), utilized for training the final model in this study—after the removal of physiological anomalies and feature engineering—is provided within the dataset/ directory.

Authors
Kailla Salsabila - Department of Informatics Engineering, Institut Teknologi Garut.

Zaki Muhamad - Department of Informatics Engineering, Institut Teknologi Garut.

Academic Integrity
All computational artifacts developed in this repository are published openly, complying with the principles of experimental reproducibility and scientific transparency in intelligent software engineering.
