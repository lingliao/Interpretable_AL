from imports import *

def main():
    # Load the CSV files
    file_path = "/content/gossis-1-eicu-only-model-ready.csv"  # specify your file path
    df_1 = pd.read_csv(file_path)

    # Check and remove rows with missing values
    df_clean = df_1.dropna()
    df_1 = df_clean.copy()

    file_path_2 = "/content/gossis-1-eicu-only.csv"  # Replace with your actual file path
    df_2 = pd.read_csv(file_path_2)

    # Find column names in df_1 that end with '_avg'
    avg_columns = [col for col in df_1.columns if col.endswith('_avg')]

    # Initialize dictionaries to store corresponding _min and _max column names
    min_columns = {}
    max_columns = {}

    # Loop through each _avg column to find corresponding _min and _max columns in df_2
    for avg_col in avg_columns:
        base_col_name = avg_col[:-4]  # Remove the '_avg' suffix
        min_col = base_col_name + '_min'
        max_col = base_col_name + '_max'

        if min_col in df_2.columns and max_col in df_2.columns:
            min_columns[avg_col] = min_col
            max_columns[avg_col] = max_col
        else:
            print(f"Corresponding columns for {avg_col} not found in df_2")

    # # Print the result
    # print(f"Number of _avg columns found: {len(avg_columns)}")

    for avg_col in avg_columns:
        if avg_col in min_columns and avg_col in max_columns:
            # print(f"{avg_col} has corresponding columns: {min_columns[avg_col]} and {max_columns[avg_col]}")
            continue
        else:
            print(f"{avg_col} does not have corresponding _min and _max columns in df_2")

    # List of columns to merge from df_2
    columns_to_merge = ['patientunitstayid'] + list(min_columns.values()) + list(max_columns.values())

    # Merge the relevant columns from df_2 to df_1 based on 'patientunitstayid'
    df_merged = df_1.merge(df_2[columns_to_merge], on='patientunitstayid', how='left')

    # Find common columns
    common_columns = df_2.columns.intersection(df_merged.columns)

    # Add additional columns
    additional_columns = ['ventilated_apache', 'apache_3j_diagnosis', 'apache_2_diagnosis',
                          'gcs_eyes_apache', 'gcs_motor_apache', 'gcs_unable_apache', 'gcs_verbal_apache']
    common_columns = common_columns.union(additional_columns)

    # Extract these columns from df_2
    df_extracted = df_2[common_columns]

    # Drop NA and columns not needed
    df_clean_new = df_extracted.dropna(subset=common_columns)
    df_clean_new = df_clean_new.drop(['patientunitstayid', 'encounter_id', 'hospital_death'], axis=1)
    df_cleaned = df_clean_new.copy()

    # Find categorical features
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns

    # Convert to numeric data
    df_encoded = df_cleaned.copy()
    for col in categorical_cols:
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

    # Select feature and target
    target = "icu_death"  # label
    features = [col for col in df_encoded.columns if col != target]  # input features

    # Extract features and target
    X = df_encoded[features].values  # extract values for model training
    y = df_encoded[target].values    # same

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Feature names
    feature_names = df_encoded[features].columns.tolist()

    # # Save your clean data and feature names if needed
    # df_encoded.to_csv('/content/cleaned_data.csv', index=False)
    # with open('/content/feature_names.txt', 'w') as f:
    #     for item in feature_names:
    #         f.write("%s\n" % item)

    # print("Data preprocessing complete. Cleaned data saved to '/content/cleaned_data.csv' and feature names saved to '/content/feature_names.txt'.")

if __name__ == "__main__":
    main()



