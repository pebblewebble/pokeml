# --- START OF (Modified) train_action_predictor_v4.py ---

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, top_k_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Input # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
import lightgbm as lgb
import argparse
import os
import joblib
import warnings
import gc # Garbage collector

# Suppress TensorFlow/warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


# --- TF Training Function (Keep as is, but ensure args match) ---
def train_tensorflow_action_predictor(X_train_processed, X_val_processed, X_test_processed,
                                      y_train_encoded, y_val_encoded, y_test_encoded, # INTEGER encoded y
                                      num_classes, class_weight_dict, label_encoder, # Pass num_classes, weights, encoder
                                      epochs=20, batch_size=128, learning_rate=0.001):
    print(f"\n--- Training TensorFlow Model ---")
    print(f"Input shape: {X_train_processed.shape[1]}")
    print(f"Num classes: {num_classes}")

    # Convert sparse matrices from ColumnTransformer to dense for TF if needed, handle potential errors
    # Or use tf.sparse.SparseTensor if model supports it
    print("Converting data to dense format for TensorFlow (may take time/memory)...")
    try:
        # Check if already dense (e.g., from direct numpy array)
        if not isinstance(X_train_processed, np.ndarray):
            X_train_processed = X_train_processed.toarray()
        if not isinstance(X_val_processed, np.ndarray):
            X_val_processed = X_val_processed.toarray()
        if not isinstance(X_test_processed, np.ndarray):
            X_test_processed = X_test_processed.toarray()
        gc.collect() # Explicitly collect garbage after large conversions
        print("Data converted to dense format.")
    except MemoryError:
        print("MemoryError: Failed to convert sparse matrix to dense. Consider reducing data size, using more memory, or a model that handles sparse input directly.")
        return None, None
    except Exception as e:
        print(f"Error converting data to dense: {e}")
        return None, None


    # Convert integer labels to one-hot encoding for categorical_crossentropy
    print("One-hot encoding target variable for TF...")
    try:
        y_train_one_hot = to_categorical(y_train_encoded, num_classes=num_classes)
        y_val_one_hot = to_categorical(y_val_encoded, num_classes=num_classes)
        y_test_one_hot = to_categorical(y_test_encoded, num_classes=num_classes)
        print("One-hot encoding complete.")
    except ValueError as e:
         print(f"Error during one-hot encoding: {e}")
         print(f"Check if y values (min={np.min(y_train_encoded)}, max={np.max(y_train_encoded)}) exceed num_classes ({num_classes}).")
         return None, None


    # --- Define the Model ---
    model = Sequential([
        Input(shape=(X_train_processed.shape[1],)), # Use Input layer for explicit shape
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax') # Softmax for multi-class probabilities
    ])

    optimizer = Adam(learning_rate=learning_rate)
    # Use categorical_crossentropy for one-hot encoded labels
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')])

    model.summary()

    # --- Train the Model ---
    print("\nStarting TF model training...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    history = model.fit(
        X_train_processed, y_train_one_hot,
        validation_data=(X_val_processed, y_val_one_hot),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=[early_stopping],
        verbose=2 # Print one line per epoch
    )
    print("TF Training finished.")

    # --- Evaluate the Model ---
    print("\nEvaluating TF model on the test set...")
    results = model.evaluate(X_test_processed, y_test_one_hot, verbose=0)
    loss = results[0]
    accuracy = results[1]
    top_5_accuracy = results[2] if len(results) > 2 else np.nan # Handle cases where metric might not be present

    print(f"TF Test Loss: {loss:.4f}")
    print(f"TF Test Accuracy: {accuracy:.4f}") # How often the single top prediction is correct
    print(f"TF Test Top-5 Accuracy: {top_5_accuracy:.4f}") # How often correct action is in top 5

    # --- Classification Report (Optional - can be slow/memory intensive) ---
    # print("\nTF Classification Report (Sample - Test Set):")
    # try:
    #     y_pred_proba = model.predict(X_test_processed)
    #     y_pred_encoded = np.argmax(y_pred_proba, axis=1) # Get class index with highest probability
    #     print(classification_report(y_test_encoded, y_pred_encoded, labels=np.arange(num_classes), target_names=label_encoder.classes_, zero_division=0))
    # except MemoryError:
    #      print("Skipping classification report due to memory constraints.")
    # except Exception as e:
    #     print(f"Could not generate classification report: {e}")

    # --- Save Model ---
    model_save_path = 'action_tf_model_v4.keras' # Use .keras format
    print(f"Saving TF model to {model_save_path}")
    try:
        model.save(model_save_path)
        print("TF Model saved.")
    except Exception as e:
        print(f"Error saving TF model: {e}")

    return history, model

# --- LGBM Training Function (Keep as is, but ensure args match) ---
def train_lgbm_action_predictor(X_train, X_val, X_test,
                                y_train_encoded, y_val_encoded, y_test_encoded, # Use INTEGER encoded y
                                numerical_features, categorical_features,
                                num_classes, class_weight_dict, label_encoder): # Pass num_classes, weights, encoder

    print(f"\n--- Training LightGBM Model ---")
    print(f"Using {len(numerical_features)} numerical and {len(categorical_features)} categorical features for LGBM.")

    # --- Preprocessing Specific to LGBM ---
    # 1. Handle Categorical Features: Convert to 'category' dtype if not already
    print("Converting categorical features to 'category' dtype for LGBM...")
    category_map = {} # To store mappings for prediction later
    X_train_lgbm = X_train.copy()
    X_val_lgbm = X_val.copy()
    X_test_lgbm = X_test.copy()

    active_categorical_features = [] # Keep track of features actually present
    for col in categorical_features:
        if col in X_train_lgbm.columns:
            active_categorical_features.append(col)
            # Combine all unique values from train, val, test to ensure consistency
            # Convert to string first to handle potential mixed types before finding unique
            all_categories = pd.concat([
                X_train_lgbm[col].astype(str),
                X_val_lgbm[col].astype(str),
                X_test_lgbm[col].astype(str)
            ]).unique()
            # Create dtype with all found categories
            cat_type = pd.CategoricalDtype(categories=all_categories, ordered=False)

            # Apply the consistent dtype
            X_train_lgbm[col] = X_train_lgbm[col].astype(str).astype(cat_type)
            X_val_lgbm[col] = X_val_lgbm[col].astype(str).astype(cat_type)
            X_test_lgbm[col] = X_test_lgbm[col].astype(str).astype(cat_type)
            category_map[col] = cat_type # Store the dtype for potential use during prediction loading
        else:
             print(f"Warning: Categorical feature '{col}' not found in training data columns for LGBM.")

    # 2. Scale Numerical Features (Optional but recommended for some objectives)
    scaler = None
    active_numerical_features = [] # Keep track of features actually present
    if numerical_features:
        print("Scaling numerical features for LGBM...")
        scaler = StandardScaler()
        # Ensure features exist before trying to scale
        active_numerical_features = [f for f in numerical_features if f in X_train_lgbm.columns]
        if active_numerical_features:
            X_train_lgbm[active_numerical_features] = scaler.fit_transform(X_train_lgbm[active_numerical_features])
            X_val_lgbm[active_numerical_features] = scaler.transform(X_val_lgbm[active_numerical_features])
            X_test_lgbm[active_numerical_features] = scaler.transform(X_test_lgbm[active_numerical_features])
            print("Numerical scaling complete.")
        else:
             print("No valid numerical features found in training data to scale for LGBM.")
    else:
        print("No numerical features defined to scale for LGBM.")

    # --- Prepare LGBM Datasets ---
    # Ensure feature names match the columns available after potential filtering
    final_feature_names = active_numerical_features + active_categorical_features
    # Reorder columns for consistency if needed, though LGBM dataset takes feature_name list
    X_train_lgbm = X_train_lgbm[final_feature_names]
    X_val_lgbm = X_val_lgbm[final_feature_names]
    X_test_lgbm = X_test_lgbm[final_feature_names]

    print("Creating LGBM datasets...")
    lgb_train = lgb.Dataset(X_train_lgbm, label=y_train_encoded,
                            categorical_feature=active_categorical_features if active_categorical_features else 'auto',
                            feature_name=final_feature_names,
                            free_raw_data=False) # Keep raw data for evaluation
    lgb_eval = lgb.Dataset(X_val_lgbm, label=y_val_encoded, reference=lgb_train,
                           categorical_feature=active_categorical_features if active_categorical_features else 'auto',
                           feature_name=final_feature_names,
                           free_raw_data=False)

    # --- Class Weights for LGBM ---
    sample_weight = None
    if class_weight_dict:
        print("Calculating sample weights for LGBM...")
        try:
             sample_weight = np.array([class_weight_dict.get(cls_idx, 1.0) for cls_idx in y_train_encoded])
             print(f"Sample weights calculated (min: {np.min(sample_weight):.2f}, max: {np.max(sample_weight):.2f}).")
             lgb_train.set_weight(sample_weight)
             print("Applied sample weights to training dataset.")
        except Exception as e:
             print(f"Warning: Could not compute or apply sample weights for LGBM: {e}.")
             sample_weight = None # Ensure it's None if setting failed

    # --- Define LGBM Parameters ---
    params = {
        'objective': 'multiclass',
        'metric': ['multi_logloss', 'multi_error'], # Logloss and classification error
        'num_class': num_classes,
        'boosting_type': 'gbdt',
        'n_estimators': 1000, # Adjusted N estimators
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'seed': 42,
        'n_jobs': -1,
        'verbose': -1,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
    }

    # --- Train LightGBM Model ---
    print("Starting LGBM model training...")
    # --- FIX: Use lgb.record_evaluation callback ---
    evals_result = {} # Dictionary to store evaluation results
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50),
        lgb.record_evaluation(evals_result) # Add the callback here
    ]
    # ---------------------------------------------

    lgbm_model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        valid_names=['train', 'eval'],
        callbacks=callbacks # Pass the list of callbacks
        # REMOVED: evals_result=evals_result
    )
    print("LGBM Training finished.")

    # --- Access Evaluation Results (Optional Example) ---
    # print("\nLGBM Evaluation Results History (Example):")
    # if 'eval' in evals_result and 'multi_logloss' in evals_result['eval']:
    #     print(f"  Eval LogLoss: {evals_result['eval']['multi_logloss'][:5]}...") # Show first few
    # if 'eval' in evals_result and 'multi_error' in evals_result['eval']:
    #     print(f"  Eval Error: {evals_result['eval']['multi_error'][:5]}...")

    # --- Evaluate LightGBM Model ---
    print("\nEvaluating LGBM model on the test set...")
    # Predict probabilities using the best iteration found by early stopping
    y_pred_proba = lgbm_model.predict(X_test_lgbm, num_iteration=lgbm_model.best_iteration)
    y_pred_indices = np.argmax(y_pred_proba, axis=1) # Get class index with highest probability

    accuracy = accuracy_score(y_test_encoded, y_pred_indices)
    try:
        top_5_accuracy = top_k_accuracy_score(y_test_encoded, y_pred_proba, k=5, labels=np.arange(num_classes))
    except ValueError:
        print("Warning: Cannot calculate top-5 accuracy (likely fewer than 5 classes in test set or predictions).")
        top_5_accuracy = np.nan

    print(f"LGBM Test Accuracy: {accuracy:.4f}")
    print(f"LGBM Test Top-5 Accuracy: {top_5_accuracy:.4f}")

    # --- Save Model and Feature Info ---
    model_save_path = 'action_lgbm_model_v4.txt'
    print(f"Saving LGBM model to {model_save_path}")
    try:
        lgbm_model.booster_.save_model(model_save_path)
        print("LGBM Model saved.")
    except Exception as e:
        print(f"Error saving LGBM model: {e}")

    lgbm_info_path = 'action_lgbm_feature_info_v4.joblib'
    print(f"Saving LGBM feature info (incl. category map) to {lgbm_info_path}")
    try:
        lgbm_info = {
            'numerical_features': active_numerical_features, # Save active features
            'categorical_features': active_categorical_features,
            'feature_names_in_order': final_feature_names, # Order used for training
            'category_map': category_map # Store the category dtypes
        }
        joblib.dump(lgbm_info, lgbm_info_path)
        print(f"LGBM feature info saved.")
    except Exception as e:
         print(f"Error saving LGBM feature info: {e}")


    # Save the scaler if it was used
    scaler_path = 'action_lgbm_scaler_v4.joblib'
    if scaler and active_numerical_features: # Check if scaler exists and was used
        print(f"Saving LGBM scaler to {scaler_path}")
        try:
             joblib.dump(scaler, scaler_path)
             print(f"LGBM scaler saved.")
        except Exception as e:
             print(f"Error saving LGBM scaler: {e}")

    return lgbm_model

# --- MODIFIED Main execution function ---
def run_action_training(parquet_path, model_type='tensorflow', min_turn=0, test_split_size=0.2, val_split_size=0.15, epochs=30, batch_size=256, learning_rate=0.001):
    """Loads data, splits, preprocesses (incl. revealed moves), and trains action predictor."""

    print(f"--- Starting Action Predictor Training (V4) ---")
    print(f"Model type: {model_type.upper()}")
    print(f"Loading data from: {parquet_path}")
    try:
        df = pd.read_parquet(parquet_path)
        print(f"Original data shape: {df.shape}")
        # --- Verify new columns exist ---
        required_cols = ['last_move_p1', 'last_move_p2', 'action_taken']
        revealed_move_cols = [col for col in df.columns if '_revealed_moves' in col]
        if not revealed_move_cols:
             print("Error: No '*_revealed_moves' columns found. Please run the updated parser.")
             return
        required_cols.extend(revealed_move_cols[:min(5, len(revealed_move_cols))]) # Check a few revealed cols

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: Required columns missing from Parquet file: {missing_cols}. Please re-run the updated parser.")
            return
        print("Required columns verified.")
        # --------------------------------
    except Exception as e:
        print(f"Error loading Parquet file: {e}"); return

    # --- Filter Data ---
    print("\nFiltering data...")
    original_rows = len(df)
    # Keep only rows with an action_taken label
    df = df.dropna(subset=['action_taken'])
    print(f"Rows after removing missing 'action_taken': {len(df)} (Removed {original_rows - len(df)})")

    if min_turn > 0:
        initial_rows_turn_filter = len(df)
        df = df[df['turn_number'] >= min_turn].copy() # Use copy to avoid SettingWithCopyWarning
        print(f"Rows after filtering turns >= {min_turn}: {len(df)} (Removed {initial_rows_turn_filter - len(df)})")
        if df.empty: print("Error: No data remaining after turn filtering."); return

    # --- Prepare Features (X) and Target (y) ---
    print("\nPreparing features and target...")
    y_raw = df['action_taken']

    # Exclude IDs, target, winner, and potentially fainted status (as it's often redundant with HP=0)
    base_exclude = ['replay_id', 'action_taken', 'battle_winner']
    # fainted_cols_to_exclude = [col for col in df.columns if '_is_fainted' in col] # Option: exclude fainted
    # cols_to_exclude = base_exclude + fainted_cols_to_exclude
    cols_to_exclude = base_exclude

    feature_columns = [col for col in df.columns if col not in cols_to_exclude]
    X = df[feature_columns].copy() # Use copy for modifications
    print(f"Initial feature count: {len(feature_columns)}")

    # --- Encode Target Variable (y) ---
    print("\nEncoding target variable (action_taken)...")
    label_encoder = LabelEncoder()
    try:
        y_encoded = label_encoder.fit_transform(y_raw)
    except TypeError as e:
        print(f"Error encoding target: {e}. Check 'action_taken' column for non-string types.")
        # Example: print(y_raw[~y_raw.apply(lambda x: isinstance(x, str))].unique())
        return
    num_classes = len(label_encoder.classes_)
    print(f"Found {num_classes} unique actions.")
    if num_classes < 2: print("Error: Need at least 2 unique actions for classification."); return
    # Save the label encoder
    label_encoder_path = 'action_label_encoder_v4.joblib'
    joblib.dump(label_encoder, label_encoder_path)
    print(f"Label encoder saved to {label_encoder_path}")

    # --- Preprocessing: Revealed Moves Multi-Hot Encoding ---
    print("\nProcessing 'revealed_moves' features (Multi-Hot Encoding)...")
    revealed_move_cols = sorted([col for col in X.columns if col.endswith('_revealed_moves')])
    all_revealed_moves = set()

    # Step 1: Find all unique moves across all relevant columns
    print("  Finding unique revealed moves...")
    for col in revealed_move_cols:
        # Handle potential NaN/None before splitting, default to empty string
        unique_in_col = X[col].fillna('').str.split(',').explode().unique()
        # Filter out empty strings or 'none' placeholder resulting from split
        all_revealed_moves.update(m for m in unique_in_col if m and m != 'none')

    unique_moves_list = sorted(list(all_revealed_moves))
    print(f"  Found {len(unique_moves_list)} unique revealed moves across all slots.")
    # Example: print(f"   Sample moves: {unique_moves_list[:10]}")

    new_binary_move_cols = []
    # Step 2 & 3: Create and populate new binary columns
    print("  Creating and populating binary revealed move columns...")
    for base_col in revealed_move_cols: # e.g., p1_slot1_revealed_moves
        # Fill NaNs in the source column before processing
        X[base_col] = X[base_col].fillna('none')
        # Efficiently check for move presence
        revealed_sets = X[base_col].str.split(',').apply(set) # Convert string to set for fast lookup
        for move in unique_moves_list:
            new_col_name = f"{base_col}_{move.replace(' ', '_')}" # Create valid column name
            X[new_col_name] = revealed_sets.apply(lambda move_set: 1 if move in move_set else 0)
            new_binary_move_cols.append(new_col_name)
        del revealed_sets # Free memory
        gc.collect()

    print(f"  Created {len(new_binary_move_cols)} new binary move features.")

    # Step 4: Drop original revealed_moves string columns
    print("  Dropping original revealed_moves string columns...")
    X = X.drop(columns=revealed_move_cols)
    gc.collect()

    # --- Identify Final Feature Types and Handle NaNs ---
    print("\nIdentifying final feature types and handling remaining NaNs...")

    # Handle 'player_to_move' (Categorical)
    if 'player_to_move' in X.columns:
        X['player_to_move'] = X['player_to_move'].fillna('unknown').astype('category')

    # Handle 'last_move_*' (Categorical)
    last_move_cols = []
    if 'last_move_p1' in X.columns:
         X['last_move_p1'] = X['last_move_p1'].fillna('none').astype('category')
         last_move_cols.append('last_move_p1')
    if 'last_move_p2' in X.columns:
         X['last_move_p2'] = X['last_move_p2'].fillna('none').astype('category')
         last_move_cols.append('last_move_p2')

    # Identify remaining categorical features (object dtype)
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_features:
        X[col] = X[col].fillna('unknown').astype('category') # Fill NaNs and convert

    # Combine all identified categorical features
    categorical_features.extend(X.select_dtypes(include=['category']).columns.tolist())
    categorical_features = sorted(list(set(categorical_features))) # Unique and sorted

    # Numerical features are everything else (including the new binary move columns)
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()

    # Ensure no overlap between numerical and categorical lists
    overlap = set(numerical_features) & set(categorical_features)
    if overlap:
        print(f"Warning: Overlap detected between numerical and categorical features: {overlap}")
        # Decide how to handle: e.g., remove from numerical if intended as categorical
        numerical_features = [f for f in numerical_features if f not in overlap]


    # Handle NaNs in numerical features (using median)
    if numerical_features:
         print("Handling NaNs in numerical features...")
         nan_counts = X[numerical_features].isnull().sum()
         cols_with_nan = nan_counts[nan_counts > 0].index.tolist()
         if cols_with_nan:
             print(f"  Numerical columns have NaNs: {cols_with_nan}. Filling with median.")
             for col in cols_with_nan:
                 median_val = X[col].median()
                 X[col] = X[col].fillna(median_val)
                 # print(f"  Filled NaNs in '{col}' with {median_val}") # Verbose
         else:
             print("  No NaNs found in numerical features.")

    # Final check for any remaining NaNs
    if X.isnull().sum().sum() > 0:
        print("Warning: NaNs still present after handling. Columns:")
        print(X.isnull().sum()[X.isnull().sum() > 0])
        print("Attempting to fill remaining NaNs with 0 (numeric) or 'unknown' (object)...")
        for col in X.columns:
             if X[col].isnull().any():
                  if pd.api.types.is_numeric_dtype(X[col]): X[col].fillna(0, inplace=True)
                  else: X[col].fillna('unknown', inplace=True)


    print(f"\nFinal feature counts after preprocessing:")
    print(f"  Numerical: {len(numerical_features)}")
    print(f"  Categorical: {len(categorical_features)}")
    print(f"  Total Features in X: {X.shape[1]}")


    # --- Split Data ---
    print("\nSplitting data into Train, Validation, Test sets...")
    try:
        # Stratify if possible (helps with imbalanced classes)
        X_train_full, X_test, y_train_full_encoded, y_test_encoded = train_test_split(
            X, y_encoded, test_size=test_split_size, random_state=42, stratify=y_encoded
        )
        # Adjust validation size based on the remaining training data
        val_size_relative = val_split_size / (1 - test_split_size)
        X_train, X_val, y_train_encoded, y_val_encoded = train_test_split(
            X_train_full, y_train_full_encoded, test_size=val_size_relative, random_state=42, stratify=y_train_full_encoded
        )
        print("Successfully split data WITH stratification.")
    except ValueError as e:
        # Stratification can fail if a class appears only once or fewer times than splits
        print(f"Warning: Stratification failed ({e}). Splitting data WITHOUT stratification.")
        X_train_full, X_test, y_train_full_encoded, y_test_encoded = train_test_split(
            X, y_encoded, test_size=test_split_size, random_state=42 # No stratify
        )
        val_size_relative = val_split_size / (1 - test_split_size)
        X_train, X_val, y_train_encoded, y_val_encoded = train_test_split(
            X_train_full, y_train_full_encoded, test_size=val_size_relative, random_state=42 # No stratify
        )
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")
    del X, df, X_train_full, y_train_full_encoded # Free memory
    gc.collect()


    # --- Calculate Class Weights ---
    print("\nCalculating class weights for handling imbalance...")
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_encoded), y=y_train_encoded)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class weights calculated (Example: {list(class_weight_dict.items())[:5]}...)")


    # --- Preprocess X data based on model type (ColumnTransformer) ---
    X_train_processed, X_val_processed, X_test_processed = None, None, None
    preprocessor = None # Define preprocessor outside conditional block

    # Save feature lists (now includes binary move features in numerical)
    feature_lists_path = 'action_feature_lists_v4.joblib'
    try:
         joblib.dump({
             'feature_columns_final': X_train.columns.tolist(), # Final columns before transformation
             'numerical_features': numerical_features,
             'categorical_features': categorical_features
             }, feature_lists_path)
         print(f"Final feature lists saved to {feature_lists_path}")
    except Exception as e:
         print(f"Error saving feature lists: {e}")


    if model_type == 'tensorflow':
        print("\nSetting up TF preprocessing pipeline (OneHotEncoder + Scaler)...")
        preprocessor_path = 'action_tf_preprocessor_v4.joblib'

        # Define transformers
        transformers = []
        if numerical_features:
             # Ensure all numerical columns actually exist in X_train
             valid_num_features = [f for f in numerical_features if f in X_train.columns]
             if valid_num_features:
                  numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
                  transformers.append(('num', numerical_transformer, valid_num_features))
             else: print("Warning: No valid numerical features found in training data for TF scaler.")
        else: print("No numerical features defined for TF scaler.")

        if categorical_features:
             # Ensure all categorical columns actually exist in X_train
             valid_cat_features = [f for f in categorical_features if f in X_train.columns]
             if valid_cat_features:
                  # Use sparse output for memory efficiency
                  categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))])
                  transformers.append(('cat', categorical_transformer, valid_cat_features))
             else: print("Warning: No valid categorical features found in training data for TF one-hot encoder.")
        else: print("No categorical features defined for TF one-hot encoder.")


        if not transformers:
            print("Error: No valid features found to build the preprocessor!"); return

        # Create and fit the preprocessor
        preprocessor = ColumnTransformer(transformers=transformers, remainder='drop') # Drop features not specified

        print("Applying TF preprocessing (fit on train, transform all)...")
        try:
            X_train_processed = preprocessor.fit_transform(X_train)
            X_val_processed = preprocessor.transform(X_val)
            X_test_processed = preprocessor.transform(X_test)
            print(f"TF Processed shapes - Train: {X_train_processed.shape}, Val: {X_val_processed.shape}, Test: {X_test_processed.shape}")
            joblib.dump(preprocessor, preprocessor_path)
            print(f"TF preprocessor saved to {preprocessor_path}")
        except ValueError as e:
             print(f"ValueError during TF preprocessing: {e}")
             print("This might happen if a categorical feature has values in val/test not seen in train, despite handle_unknown='ignore'. Check data consistency.")
             # Debug: Check categories in train vs val/test for a sample column
             # sample_col = valid_cat_features[0] if valid_cat_features else None
             # if sample_col:
             #      train_cats = set(X_train[sample_col].unique())
             #      val_cats = set(X_val[sample_col].unique())
             #      test_cats = set(X_test[sample_col].unique())
             #      print(f"Categories diff Val vs Train for '{sample_col}': {val_cats - train_cats}")
             #      print(f"Categories diff Test vs Train for '{sample_col}': {test_cats - train_cats}")
             return
        except MemoryError:
            print("MemoryError during TF preprocessing (likely OneHotEncoder). Consider reducing features or using more memory.")
            return
        except Exception as e:
            print(f"Error during TF preprocessing: {e}")
            import traceback; traceback.print_exc(); return

        # Free up memory from original splits
        del X_train, X_val, X_test
        gc.collect()


    elif model_type == 'lightgbm':
        print("\nPreprocessing for LGBM (dtype conversion, scaling) will occur inside its training function.")
        # Assign data directly, preprocessing happens inside train_lgbm_action_predictor
        X_train_processed, X_val_processed, X_test_processed = X_train, X_val, X_test
        # Note: We pass the original X_train, X_val, X_test splits here.
        # The LGBM function will handle dtype conversion and scaling internally.
        preprocessor_path = None # No separate preprocessor object for LGBM needed here

    else:
        print(f"Error: Unknown model_type '{model_type}'"); return


    # --- Train Selected Model ---
    print(f"\n--- Initiating {model_type.upper()} Model Training ---")
    if model_type == 'tensorflow':
         if X_train_processed is not None: # Check if preprocessing succeeded
             train_tensorflow_action_predictor(X_train_processed, X_val_processed, X_test_processed,
                                               y_train_encoded, y_val_encoded, y_test_encoded,
                                               num_classes, class_weight_dict, label_encoder,
                                               epochs, batch_size, learning_rate)
         else: print("Skipping TF training due to preprocessing errors.")
    elif model_type == 'lightgbm':
         # Pass the original X splits and feature lists
         train_lgbm_action_predictor(X_train_processed, X_val_processed, X_test_processed, # These are still original DFs here
                                     y_train_encoded, y_val_encoded, y_test_encoded,
                                     numerical_features, categorical_features, # Pass final lists
                                     num_classes, class_weight_dict, label_encoder)


# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train action predictor (V4) incorporating revealed moves and last_move features.")
    parser.add_argument("parquet_file", type=str, help="Path to the input Parquet file from the updated parser.")
    parser.add_argument("--model_type", choices=['tensorflow', 'lightgbm'], default='lightgbm', help="Type of model to train (default: lightgbm).")
    parser.add_argument("--min_turn", type=int, default=1, help="Minimum turn number to include (default: 1). Set to 0 to include turn 0.")
    parser.add_argument("--test_split", type=float, default=0.2, help="Fraction of data for the test set (default: 0.2).")
    parser.add_argument("--val_split", type=float, default=0.15, help="Fraction of data for the validation set (relative to initial data, default: 0.15).")
    # TF specific args
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs (TensorFlow only).")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training (TensorFlow only).")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for Adam optimizer (TensorFlow only).")

    args = parser.parse_args()

    # Validate split sizes
    if args.test_split + args.val_split >= 1.0:
        print("Error: Sum of test_split and val_split must be less than 1.0")
        exit(1)

    run_action_training(
        parquet_path=args.parquet_file,
        model_type=args.model_type,
        min_turn=args.min_turn,
        test_split_size=args.test_split,
        val_split_size=args.val_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

    print("\nAction predictor training script (v4) finished.")
# --- END OF (Modified) train_action_predictor_v4.py ---