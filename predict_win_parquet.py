import pandas as pd
import numpy as np
import tensorflow as tf
import lightgbm as lgb
import joblib
import argparse
import os

# Suppress TensorFlow warnings (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

def run_prediction(input_parquet_path, output_parquet_path, model_path, model_type, preprocessor_path=None, scaler_path=None, features_path=None):
    """
    Loads data, applies preprocessing, runs prediction, and saves results.
    """
    print(f"--- Starting Prediction ---")
    print(f"Input Parquet: {input_parquet_path}")
    print(f"Output Parquet: {output_parquet_path}")
    print(f"Model Path: {model_path}")
    print(f"Model Type: {model_type}")
    if preprocessor_path: print(f"Preprocessor Path (TF): {preprocessor_path}")
    if scaler_path: print(f"Scaler Path (LGBM): {scaler_path}")
    if features_path: print(f"Features List Path: {features_path}")

    # --- 1. Load Input Data ---
    try:
        df_predict = pd.read_parquet(input_parquet_path)
        print(f"\nLoaded data for prediction. Shape: {df_predict.shape}")
        if df_predict.empty:
            print("Error: Input Parquet file is empty.")
            return
    except Exception as e:
        print(f"Error loading input Parquet file '{input_parquet_path}': {e}")
        return

    # --- 2. Load Features List (Crucial!) ---
    if not features_path or not os.path.exists(features_path):
        print(f"Error: Feature list file '{features_path}' not found. Cannot proceed without knowing the required features.")
        # As a fallback, you could try regenerating the feature list here using the *exact* same logic
        # as in train.py, but loading it is safer.
        # Example fallback (use with caution, ensure logic matches train.py):
        print("Attempting to regenerate feature list (ensure logic matches training!)...")
        base_exclude = ['replay_id', 'action_taken', 'battle_winner']
        fainted_cols_to_exclude = [col for col in df_predict.columns if '_is_fainted' in col]
        species_cols_to_exclude = [col for col in df_predict.columns if '_species' in col]
        revealed_cols_to_exclude = [col for col in df_predict.columns if '_revealed_moves' in col]
        cols_to_exclude = base_exclude + fainted_cols_to_exclude + species_cols_to_exclude + revealed_cols_to_exclude
        feature_columns = [col for col in df_predict.columns if col not in cols_to_exclude]
        if not feature_columns:
             print("Error: Could not regenerate feature columns.")
             return
        print(f"Regenerated {len(feature_columns)} features.")
        # return # Safer to stop if features weren't explicitly loaded

    else:
        try:
            feature_columns = joblib.load(features_path)
            print(f"Loaded {len(feature_columns)} features from '{features_path}'.")
        except Exception as e:
            print(f"Error loading feature list from '{features_path}': {e}")
            return

    # --- 3. Prepare Feature Subset (X) ---
    # Check if all required feature columns exist in the input data
    missing_cols = [col for col in feature_columns if col not in df_predict.columns]
    if missing_cols:
        print(f"Error: The input data is missing the following required feature columns: {missing_cols}")
        return

    X_predict = df_predict[feature_columns].copy()
    print(f"Prepared prediction feature set X_predict. Shape: {X_predict.shape}")

    # --- 4. Load Model and Preprocessor/Scaler ---
    model = None
    preprocessor = None
    scaler = None

    try:
        if model_type == 'tensorflow':
            if not preprocessor_path or not os.path.exists(preprocessor_path):
                print(f"Error: TensorFlow preprocessor file '{preprocessor_path}' not found.")
                return
            print(f"Loading TensorFlow model from {model_path}...")
            model = tf.keras.models.load_model(model_path)
            print(f"Loading TF preprocessor from {preprocessor_path}...")
            preprocessor = joblib.load(preprocessor_path)

        elif model_type == 'lightgbm':
            print(f"Loading LightGBM model from {model_path}...")
            model = joblib.load(model_path) # Load the saved LGBMClassifier object
            if scaler_path and os.path.exists(scaler_path):
                 print(f"Loading LGBM scaler from {scaler_path}...")
                 scaler = joblib.load(scaler_path)
            elif scaler_path:
                 print(f"Warning: LGBM scaler file '{scaler_path}' specified but not found. Numerical features will not be scaled.")
            else:
                 print("No LGBM scaler path provided or needed.")

        else:
            print(f"Error: Unknown model_type '{model_type}'")
            return
    except Exception as e:
        print(f"Error loading model or preprocessor/scaler: {e}")
        return

    print("Model and helpers loaded successfully.")

    # --- 5. Preprocess X_predict ---
    # Apply the *exact same* preprocessing steps as during training
    X_predict_processed = None

    if model_type == 'tensorflow':
        if preprocessor:
            print("Applying TF preprocessing to prediction data...")
            try:
                 # Important: Ensure the columns used by the preprocessor are present
                 # The ColumnTransformer should handle selecting the correct columns
                 # if X_predict contains only the feature_columns.
                 X_predict_processed = preprocessor.transform(X_predict)
                 print(f"TF Preprocessed prediction data shape: {X_predict_processed.shape}")
                 if X_predict_processed.shape[1] == 0:
                     print("Error: TF Preprocessing resulted in 0 features.")
                     return
                 # Check if the number of features matches the model's input layer
                 expected_features = model.input_shape[-1]
                 if X_predict_processed.shape[1] != expected_features:
                     print(f"Error: Processed feature count ({X_predict_processed.shape[1]}) does not match model input ({expected_features}). Check feature list and preprocessor.")
                     return

            except Exception as e:
                 print(f"Error during TF preprocessing for prediction: {e}")
                 # Potentially helpful debug: Check dtypes, missing values in X_predict before transform
                 # print(X_predict.info())
                 # print(X_predict.isnull().sum())
                 return
        else:
            print("Error: TF Preprocessor not loaded.")
            return

    elif model_type == 'lightgbm':
        print("Applying LGBM preprocessing to prediction data...")
        # Replicate preprocessing steps from train_lgbm_predictor

        # a) Identify categorical and numerical features within X_predict
        categorical_features_predict = X_predict.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_features_predict = X_predict.select_dtypes(include=np.number).columns.tolist()

        # b) Handle NaNs (using median from training *or* prediction set - consistency is key)
        # For simplicity, using median of current prediction set if NaNs exist
        if numerical_features_predict:
            nan_counts = X_predict[numerical_features_predict].isnull().sum()
            cols_with_nan = nan_counts[nan_counts > 0].index.tolist()
            if cols_with_nan:
                print(f"Warning: Numerical columns in prediction data have NaNs: {cols_with_nan}. Filling with median.")
                for col in cols_with_nan:
                    X_predict[col] = X_predict[col].fillna(X_predict[col].median())

        # c) Convert object columns to category (if any were object) and ensure all known cats are category dtype
        object_cols = X_predict.select_dtypes(include=['object']).columns
        for col in object_cols:
            X_predict[col] = X_predict[col].fillna('unknown').astype('category')
            if col not in categorical_features_predict:
                 categorical_features_predict.append(col) # Add to list if newly converted

        # Ensure existing category columns are still category type
        for col in categorical_features_predict:
             if col in X_predict.columns and not pd.api.types.is_categorical_dtype(X_predict[col]):
                 print(f"Converting column '{col}' back to category dtype.")
                 X_predict[col] = X_predict[col].astype('category')
             # Important: Handle potential *new* categories not seen in training.
             # This basic `astype('category')` might not be sufficient if the training
             # categories were explicitly set. A more robust approach would involve
             # loading the categories saved during training. For now, we proceed.

        # d) Scale numerical features if scaler was loaded
        if scaler and numerical_features_predict:
            print("Scaling numerical features for prediction using loaded LGBM scaler...")
            try:
                X_predict[numerical_features_predict] = scaler.transform(X_predict[numerical_features_predict])
            except Exception as e:
                print(f"Error applying LGBM scaler: {e}")
                # Check if columns match scaler's expectations
                # print(f"Scaler expects {scaler.n_features_in_} features. Provided columns: {numerical_features_predict}")
                return
        elif scaler and not numerical_features_predict:
             print("Warning: LGBM scaler loaded, but no numerical features found in prediction data.")
        elif not scaler and numerical_features_predict:
             print("Warning: Numerical features found, but no LGBM scaler loaded/found. Prediction will use unscaled features.")


        X_predict_processed = X_predict # For LGBM, the DataFrame itself is often used directly
        print(f"LGBM Preprocessed prediction data shape: {X_predict_processed.shape}")


    # --- 6. Make Predictions ---
    print("\nMaking predictions...")
    predictions = None
    try:
        if model_type == 'tensorflow':
            y_pred_proba = model.predict(X_predict_processed)
            predictions = (y_pred_proba > 0.5).astype(int).flatten() # Convert probabilities to 0 or 1

        elif model_type == 'lightgbm':
            predictions = model.predict(X_predict_processed) # Directly outputs 0 or 1

    except Exception as e:
        print(f"Error during prediction: {e}")
        # If TF error, check X_predict_processed shape/dtype
        # If LGBM error, check X_predict_processed dtypes, esp. category
        # print(X_predict_processed.info())
        return

    if predictions is None:
        print("Error: Predictions could not be generated.")
        return

    print(f"Generated {len(predictions)} predictions.")

    # --- 7. Append Predictions to DataFrame ---
    output_col_name = 'predicted_winner'
    if output_col_name in df_predict.columns:
        print(f"Warning: Column '{output_col_name}' already exists. It will be overwritten.")
    df_predict[output_col_name] = predictions
    # Optionally map 0/1 back to 'p2'/'p1'
    # df_predict[output_col_name] = df_predict[output_col_name].map({0: 'p2', 1: 'p1'})

    print(f"Predictions added to DataFrame under column '{output_col_name}'.")
    print("\nSample of data with predictions:")
    print(df_predict[[col for col in df_predict.columns if col not in feature_columns] + [output_col_name]].head()) # Show non-feature cols + prediction

    # --- 8. Save Output Parquet ---
    try:
        print(f"\nSaving results to {output_parquet_path}...")
        df_predict.to_parquet(output_parquet_path, index=False)
        print("Prediction results saved successfully.")
    except Exception as e:
        print(f"Error saving output Parquet file '{output_parquet_path}': {e}")

    print("--- Prediction Finished ---")


# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Pokemon battle winner prediction.")
    parser.add_argument("input_parquet", type=str, help="Path to the input Parquet file with game states.")
    parser.add_argument("output_parquet", type=str, help="Path to save the output Parquet file with predictions.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file (.keras for TF, .joblib for LGBM).")
    parser.add_argument("--model_type", choices=['tensorflow', 'lightgbm'], required=True, help="Type of the loaded model.")
    parser.add_argument("--preprocessor_path", type=str, default='tf_preprocessor.joblib', help="Path to the saved TensorFlow preprocessor (ColumnTransformer joblib file). Required if model_type is tensorflow.")
    parser.add_argument("--scaler_path", type=str, default='lgbm_scaler.joblib', help="Path to the saved LightGBM scaler (StandardScaler joblib file). Optional for lightgbm.")
    parser.add_argument("--features_path", type=str, default='feature_list.joblib', help="Path to the saved list of feature column names (joblib file).")

    args = parser.parse_args()

    # Basic validation
    if args.model_type == 'tensorflow' and not args.preprocessor_path:
        parser.error("--preprocessor_path is required when --model_type is tensorflow")
    if not args.features_path:
         print("Warning: --features_path not specified. Attempting to regenerate features, but loading is recommended for consistency.")
         # Allow proceeding but with the warning


    run_prediction(
        input_parquet_path=args.input_parquet,
        output_parquet_path=args.output_parquet,
        model_path=args.model_path,
        model_type=args.model_type,
        preprocessor_path=args.preprocessor_path if args.model_type == 'tensorflow' else None,
        scaler_path=args.scaler_path if args.model_type == 'lightgbm' else None,
        features_path=args.features_path
    )