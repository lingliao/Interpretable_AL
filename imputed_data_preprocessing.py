from imports import *

def main():
    file_path = "/content/gossis-1-eicu-only-model-ready.csv"  # specify your file path
    df_1 = pd.read_csv(file_path)

    if df_1.isnull().values.any():  # check if any missing values exist
        df_clean = df_1.dropna()  # remove the rows with missing items
    else:
        df_clean = df_1

    if not df_clean.isnull().values.any():  # check if the previous step removed all missing rows
        df_cleaned = df_clean.drop(['patientunitstayid', 'encounter_id', 'partition', 'hospital_death'], axis=1)
    
    print(df_cleaned.shape)

    # 1. Find categorical columns
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns

    # 2. Convert categorical variables to numeric data
    df_encoded = df_cleaned.copy()  # Create a copy of the DataFrame to avoid modifying the original data
    for col in categorical_cols:
        # Apply LabelEncoder to each categorical column
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

    # 3. Select features and target
    target = "icu_death"  # Specify the target label
    # Use list comprehension to select all columns except the target as features
    features = [col for col in df_encoded.columns if col != target]

    # 4. Extract feature and target values
    X = df_encoded[features].values  # Extract the values for input features, used for model training
    y = df_encoded[target].values  # Extract the values for the target label

    # 5. Normalize the feature data
    scaler = StandardScaler()  # Initialize the StandardScaler
    X = scaler.fit_transform(X)  # Apply standardization to the feature data

    # 6. Feature name 
    feature_names = df_encoded[features].columns.tolist()

    # # Save the processed data to a CSV file (optional)
    # processed_data = pd.concat([pd.DataFrame(X, columns=features), pd.DataFrame(y, columns=[target])], axis=1)
    # processed_data.to_csv("/content/processed_data.csv", index=False)

if __name__ == "__main__":
    main()