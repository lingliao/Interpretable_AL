# Filename: imports.py

# Import essential libraries
import os              # OS module for operating system dependent functionality
import time            # Time module for time related functions
import joblib          # Joblib for efficient serialization of large numpy arrays
import pandas as pd    # Pandas for data manipulation and analysis
import numpy as np     # Numpy for numerical operations
import matplotlib.pyplot as plt  # Matplotlib for plotting and visualization
from tqdm import tqdm  # Tqdm for progress bar indication

# Import sklearn modules for model building, preprocessing, and evaluation
from sklearn.model_selection import train_test_split, StratifiedKFold  # For training-testing split and k-fold cross-validation
from sklearn.preprocessing import LabelEncoder, StandardScaler         # For encoding labels and scaling features
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # RandomForest models for classification and regression

# Import sklearn metrics for performance evaluation
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, precision_recall_curve

# Import XGBoost for gradient boosting models
from xgboost import XGBClassifier  # XGBoost classifier

# Import SHAP for model interpretability
import shap  # SHAP for explaining model predictions

# Import KMeans clustering algorithm
from sklearn.cluster import KMeans  # KMeans algorithm for clustering

# Example usage of KMeans for clustering would be added here if needed

# Note:
# Ensure SHAP is installed in your working environment using:
# pip install shap