from imports import *

def plot_model_performance(model_name, X, y, n_splits=5):
    """Plot performance metrics across all folds for a given model using the correct validation splits."""
    # Initialize lists to store metrics for all folds
    all_fpr = []
    all_tpr = []
    all_precision = []
    all_recall = []
    all_accuracy = []
    all_auroc = []
    all_auprc = []
    fold_labels = []

    # Initialize cross-validation splitter
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Base folder where models are saved
    save_folder = f'saved_models_{model_name}'

    # Iterate over each fold
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        # Load model for the current fold
        if model_name == 'xgboost':  # Load XGBoost models
            model_path = os.path.join(save_folder, f'{model_name}_fold_{fold+1}.json')
            model = XGBClassifier()
            model.load_model(model_path)
        else:  # Load other models (e.g., RandomForest) serialized with joblib
            model_path = os.path.join(save_folder, f'{model_name}_fold_{fold+1}.pkl')
            model = joblib.load(model_path)

        # Get the correct validation data for this fold
        X_val = X[val_idx]
        y_val = y[val_idx]

        # Get prediction probabilities
        y_pred = model.predict_proba(X_val)

        # Assuming binary classification
        if y_pred.shape[1] == 2:
            y_pred = y_pred[:, 1]

        # Compute metrics
        fpr, tpr, _ = roc_curve(y_val, y_pred)
        precision, recall, _ = precision_recall_curve(y_val, y_pred)
        accuracy = accuracy_score(y_val, (y_pred > 0.5).astype(int))
        auroc = roc_auc_score(y_val, y_pred)
        auprc = average_precision_score(y_val, y_pred)

        # Store metrics for this fold
        all_fpr.append(fpr)
        all_tpr.append(tpr)
        all_precision.append(precision)
        all_recall.append(recall)
        all_accuracy.append(accuracy)
        all_auroc.append(auroc)
        all_auprc.append(auprc)
        fold_labels.append(f'Fold {fold+1} (Acc={accuracy:.2f})')

    # Create figure with two subplots
    plt.figure(figsize=(16, 6))

    # Plot ROC curves
    plt.subplot(1, 2, 1)
    for i in range(n_splits):
        plt.plot(all_fpr[i], all_tpr[i],
                 label=f'{fold_labels[i]}\nAUROC = {all_auroc[i]:.3f}',
                 linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{model_name.upper()} - ROC Curves (Across {n_splits} Folds)', fontsize=14, pad=20)

    # Add mean AUROC to the plot
    mean_auroc = np.mean(all_auroc)
    std_auroc = np.std(all_auroc)
    plt.text(0.4, 0.05,
             f'Mean AUROC = {mean_auroc:.3f} ± {std_auroc:.3f}',
             bbox=dict(facecolor='white', alpha=0.8),
             fontsize=12)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)

    # Plot PR curves
    plt.subplot(1, 2, 2)
    for i in range(n_splits):
        plt.plot(all_recall[i], all_precision[i],
                 label=f'{fold_labels[i]}\nAUPRC = {all_auprc[i]:.3f}',
                 linewidth=2)

    # Add baseline (positive class ratio)
    positive_ratio = np.mean(y)  # Calculate the positive class ratio dynamically
    plt.axhline(y=positive_ratio, color='gray', linestyle='--', alpha=0.5,
                label=f'Baseline (Pos Ratio = {positive_ratio:.4f})')

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'{model_name.upper()} - Precision-Recall Curves (Across {n_splits} Folds)', fontsize=14, pad=20)

    # Add mean AUPRC to the plot
    mean_auprc = np.mean(all_auprc)
    std_auprc = np.std(all_auprc)
    plt.text(0.4, 0.05,
             f'Mean AUPRC = {mean_auprc:.3f} ± {std_auprc:.3f}',
             bbox=dict(facecolor='white', alpha=0.8),
             fontsize=12)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10) # this is adjustable as well
    plt.grid(True, alpha=0.3)

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f'{model_name}_performance_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":

    # Example usage
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    # Assuming that X and y have already been preprocessed and are available here.
    # You can replace this section with your own data loading and preprocessing logic.

    # Example data loading (adjust the file paths and column names as necessary)
    file_path = "path_to_your_data_file.csv"
    data = pd.read_csv(file_path)
    
    # Process your features and target labels accordingly
    target = "icu_death"  # adjust this to your target variable name
    categorical_cols = data.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        data[col] = LabelEncoder().fit_transform(data[col])

    features = [col for col in data.columns if col != target]
    X = data[features].values
    y = data[target].values
    
    # Normalize the feature data (optional)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Plot performance metrics for a given model
    plot_model_performance('randomforest', X, y, n_splits=5)

