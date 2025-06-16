from imports import *
from real_data_preprocessing import preprocess_data

def calculate_cluster_importance(feature_names, shap_values, num_clusters=20):
    """
    Calculate and visualize the importance of feature clusters.
    """
    # Compute mean absolute SHAP values for each feature
    mean_shap_values = np.abs(shap_values).mean(axis=0)

    # Reshape for clustering
    mean_shap_values = mean_shap_values.reshape(-1, 1)

    # Clustering features
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(mean_shap_values)
    feature_clusters = kmeans.labels_

    # Calculate cluster importance by summing SHAP values within each cluster
    cluster_importance = {}
    for cluster in range(num_clusters):
        cluster_indices = np.where(feature_clusters == cluster)[0]
        cluster_importance[cluster] = mean_shap_values[cluster_indices].sum()

    # Normalize cluster importance
    total_importance = sum(cluster_importance.values())
    for cluster in cluster_importance:
        cluster_importance[cluster] /= total_importance

    # Visualization
    cluster_names = [f'Cluster {i+1}' for i in range(num_clusters)]
    cluster_values = [cluster_importance[i] for i in range(num_clusters)]

    plt.figure(figsize=(10, 6))
    plt.barh(cluster_names, cluster_values, color='skyblue')
    plt.xlabel('Normalized Importance')
    plt.ylabel('Clusters')
    plt.title('Feature Cluster Importance')
    plt.show()

    # Print cluster details
    print("Cluster details:")
    for cluster in range(num_clusters):
        cluster_indices = np.where(feature_clusters == cluster)[0]
        print(f"\nCluster {cluster + 1} features:")
        for idx in cluster_indices:
            print(f"  {feature_names[idx]} (SHAP: {mean_shap_values[idx][0]:.4f})")

def run_cv_with_model(X, y, model_name='xgboost', feature_names=None, num_clusters=20):
    """with 5 fold cross validation"""
    print("\n" + "="*60)
    print(f"【{model_name.upper()}】starting five fold cross validation - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print("📊 original data distribution:")
    print(f"  negative: {np.sum(y == 0)} | positive: {np.sum(y == 1)}")
    print(f"  positive percentage: {np.mean(y):.2%}\n")

    # Data check
    assert len(X) == len(y), "❌ size mismatch between features and labels！"
    assert not np.isnan(y).any(), "❌ missing data in labels！"
    assert len(np.unique(y)) > 1, "❌ only one type of label！"

    # Initialize 5 fold cross validation
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # For each fold
    print("\n🔍 per fold result：")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        y_train, y_val = y[train_idx], y[val_idx]
        diff = abs(y_train.mean() - y_val.mean()) * 100
        status = "✅" if diff < 1 else "❌"
        print(f"  {status} Fold {fold + 1}: training {y_train.mean():.2%} positive | "
              f"validation {y_val.mean():.2%} positive | difference: {diff:.2f}%")

    # Initialize output lists
    auroc_scores = []
    auprc_scores = []
    training_times = []
    fold_models = []

    # For training
    print(f"\n🏋️ training begin {model_name.upper()} model...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        fold_start_time = time.time()
        print("\n" + "="*60)
        print(f"🔄 FOLD {fold + 1}/{n_splits} - {time.strftime('%H:%M:%S')}")
        print(f"  training: {len(train_idx)} | validation: {len(val_idx)}")
        print(f"  positive percentage in training: {y[train_idx].mean():.2%} | positive percentage in validation: {y[val_idx].mean():.2%}")
        print("="*60)

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if model_name == 'xgboost':
            model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42 + fold,
                use_label_encoder=False,
                eval_metric=['logloss', 'auc', 'aucpr'],
                early_stopping_rounds=10,
                verbosity=1
            )
            print("\n⚙️ XGBoost parameter:")
            print(f"  n_estimators=200 | max_depth=6 | lr=0.05")
            print(f"  early_stopping=20 | eval_metric=[logloss, auc, aucpr]")

        elif model_name == 'randomforest':
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42 + fold,
                class_weight='balanced',
                verbose=0,
                n_jobs=-1
            )
            print("\n⚙️ Random Forest parameter:")
            print(f"  n_estimators=200 | max_depth=None")
            print(f"  class_weight=balanced | n_jobs=-1")

        # Training process
        print("\n⏳ training process:")
        if model_name == 'xgboost':
            # For XGBoost
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=10  # output per 10 epochs
            )

            # Print the output
            results = model.evals_result()
            for metric in ['logloss', 'auc', 'aucpr']:
                print(f"  {metric.upper():<8} final value: "
                      f"training={results['validation_0'][metric][-1]:.4f} | "
                      f"validation={results['validation_1'][metric][-1]:.4f}")
        else:
            # Random Forest training
            print("  RF training begins")
            with tqdm(total=model.n_estimators, desc="tree building") as pbar:
                model.fit(X_train, y_train)
                pbar.update(model.n_estimators)

            # Output the tree built process
            print("\n🌲 tree built completed:")
            print(f"  number of trees built: {len(model.estimators_)}")
            print(f"  average depth of each tree: "
                  f"{np.mean([tree.tree_.max_depth for tree in model.estimators_]):.1f}")

        # Evaluate the model
        fold_time = time.time() - fold_start_time
        training_times.append(fold_time)

        preds = model.predict_proba(X_val)
        if preds.shape[1] == 2:  # binary classification
            preds = preds[:, 1]
            auroc = roc_auc_score(y_val, preds)
            auprc = average_precision_score(y_val, preds)
        else:  # multiple classification
            auroc = roc_auc_score(y_val, preds, multi_class='ovr')
            auprc = average_precision_score(y_val, preds, average='macro')

        print(f"\n🎯 Fold {fold + 1} result:")
        print(f"  ⏱️ time: {fold_time:.1f} seconds")
        print(f"  🎯 AUROC: {auroc:.4f}")
        print(f"  🎯 AUPRC: {auprc:.4f}")

        auroc_scores.append(auroc)
        auprc_scores.append(auprc)
        fold_models.append(model)

        # Model save
        save_folder = f'saved_models_{model_name}'
        os.makedirs(save_folder, exist_ok=True)

        if model_name == 'xgboost':
            model_save_path = os.path.join(save_folder, f'{model_name}_fold_{fold+1}.json')
            model.save_model(model_save_path)
        else:
            model_save_path = os.path.join(save_folder, f'{model_name}_fold_{fold+1}.pkl')
            joblib.dump(model, model_save_path)

        print(f"💾 model saved to: {model_save_path}")

        # SHAP plotting and cluster importance calculation
        if model_name in ['xgboost', 'randomforest']:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_val)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification

            # Plot SHAP summary
            plt.figure()
            shap.summary_plot(shap_values, X_val, feature_names=feature_names, show=False)
            plt.title(f'SHAP Summary Plot for Fold {fold+1}')
            plt.savefig(os.path.join(save_folder, f'shap_fold_{fold+1}.png'), dpi=300, bbox_inches='tight')
            plt.show()

            calculate_cluster_importance(feature_names, shap_values, num_clusters=num_clusters)

    # Final report
    print("\n\n" + "="*60)
    print(f"🔥 {model_name.upper()} final result - average training time: {np.mean(training_times):.1f} seconds per fold")
    print("="*60)
    print(f"📊 Avg AUROC: {np.mean(auroc_scores):.4f} ± {np.std(auroc_scores):.4f}")
    print(f"📊 Avg AUPRC: {np.mean(auprc_scores):.4f} ± {np.std(auprc_scores):.4f}")

    print("\n📝 results for each fold:")
    for fold in range(n_splits):
        print(f"  Fold {fold + 1}: "
              f"AUROC={auroc_scores[fold]:.4f} | "
              f"AUPRC={auprc_scores[fold]:.4f} | "
              f"time consumed={training_times[fold]:.1f} s")

    print(f"\n💾 all models are saved to : {save_folder}/")
    return auroc_scores, auprc_scores

if __name__ == "__main__":
    # file_path = "/content/gossis-1-eicu-only-model-ready.csv"  # specify your file path
    X, y, feature_names = preprocess_data(file_path)

    # Train the model
    run_cv_with_model(X, y, 'xgboost', feature_names=feature_names, num_clusters=20) # choose the model you plan to use



