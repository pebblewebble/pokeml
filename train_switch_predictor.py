import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, Activation # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import lightgbm as lgb
import argparse
import os
import joblib
import warnings
import gc # Garbage collector
import re # For sanitizing move names

# Suppress TensorFlow/warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Helper Function (find_active_species - unchanged) ---
def find_active_species(row, player_prefix):
    """
    Finds the species of the active Pokemon for a given player prefix ('p1' or 'p2')
    in a DataFrame row containing slot information.
    """
    for i in range(1, 7): # Check slots 1 to 6
        active_col = f"{player_prefix}_slot{i}_is_active"
        species_col = f"{player_prefix}_slot{i}_species"
        if active_col in row.index and species_col in row.index:
            if row[active_col] == 1:
                return row[species_col] if pd.notna(row[species_col]) else 'Unknown'
    return 'Unknown'

# --- TF Training Function (train_tensorflow_switch_predictor - unchanged) ---
def train_tensorflow_switch_predictor(X_train_processed, X_val_processed, X_test_processed,
                                     y_train, y_val, y_test, # Use BINARY y (0/1)
                                     class_weight_dict, # Pass weights
                                     epochs=20, batch_size=128, learning_rate=0.001,
                                     model_suffix=""): # Use model_suffix
    print(f"\n--- Training TensorFlow Switch Predictor (Binary) ---")
    print(f"Input shape: {X_train_processed.shape[1]}")
    print(f"Target is binary (0=move, 1=switch)")

    # Define the Model (Binary Output)
    input_dim = X_train_processed.shape[1]
    print(f"Building TF model with input dimension: {input_dim}")
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, use_bias=False),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.3),
        Dense(128, use_bias=False),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.3),
        Dense(64, use_bias=False),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid') # *** 1 output neuron, sigmoid activation ***
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', # *** Binary loss ***
                  metrics=['accuracy'])
    model.summary()

    # Train the Model
    print("\nStarting TF model training...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    history = model.fit(
        X_train_processed, y_train, # Pass binary y directly
        validation_data=(X_val_processed, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict, # Still use class weights for imbalance
        callbacks=[early_stopping],
        verbose=2
    )
    print("TF Training finished.")

    # Evaluate the Model
    print("\nEvaluating TF model on the test set...")
    results = model.evaluate(X_test_processed, y_test, verbose=0)
    loss = results[0]
    accuracy = results[1]
    print(f"TF Test Loss: {loss:.4f}")
    print(f"TF Test Accuracy: {accuracy:.4f}")

    # Predict probabilities for AUC
    try:
        y_pred_proba = model.predict(X_test_processed).flatten()
        auc = roc_auc_score(y_test, y_pred_proba)
        print(f"TF Test AUC: {auc:.4f}")
    except Exception as e:
        print(f"Could not calculate AUC: {e}")


    # Save Model
    model_save_path = f'switch_predictor_tf_model_v2_{model_suffix}.keras' # Adjusted name version
    print(f"Saving TF model to {model_save_path}")
    try:
        model.save(model_save_path)
        print("TF Model saved.")
    except Exception as e:
        print(f"Error saving TF model: {e}")

    return history, model


# --- LGBM Training Function (train_lgbm_switch_predictor - unchanged) ---
def train_lgbm_switch_predictor(X_train, X_val, X_test,
                                y_train, y_val, y_test, # Use BINARY y (0/1)
                                numerical_features, categorical_features,
                                class_weight_dict, # Pass weights dict (for sample weights)
                                model_suffix=""): # Use model_suffix
    print(f"\n--- Training LightGBM Switch Predictor (Binary) ---")
    print(f"Using {len(numerical_features)} numerical and {len(categorical_features)} categorical features for LGBM.")

    # --- Preprocessing Specific to LGBM ---
    print("Converting categorical features to 'category' dtype for LGBM...")
    category_map = {}
    X_train_lgbm = X_train.copy()
    X_val_lgbm = X_val.copy()
    X_test_lgbm = X_test.copy()
    active_categorical_features = []
    for col in categorical_features:
        if col in X_train_lgbm.columns:
            active_categorical_features.append(col)
            # Combine categories from all splits to ensure consistency
            all_categories = pd.concat([
                X_train_lgbm[col].astype(str).fillna('Unknown'), # Handle potential NaNs introduced before split
                X_val_lgbm[col].astype(str).fillna('Unknown'),
                X_test_lgbm[col].astype(str).fillna('Unknown')
            ]).unique()
            # Filter out potential numeric types misinterpreted as categories if necessary
            all_categories = [cat for cat in all_categories if isinstance(cat, str)]
            cat_type = pd.CategoricalDtype(categories=sorted(list(all_categories)), ordered=False)
            try:
                X_train_lgbm[col] = X_train_lgbm[col].astype(str).fillna('Unknown').astype(cat_type)
                X_val_lgbm[col] = X_val_lgbm[col].astype(str).fillna('Unknown').astype(cat_type)
                X_test_lgbm[col] = X_test_lgbm[col].astype(str).fillna('Unknown').astype(cat_type)
                category_map[col] = cat_type
            except ValueError as e:
                 print(f"Warning: Issue setting category dtype for {col}: {e}. Using default category conversion.")
                 # Fallback if strict dtype fails (e.g., unexpected values)
                 X_train_lgbm[col] = X_train_lgbm[col].astype(str).fillna('Unknown').astype('category')
                 X_val_lgbm[col] = X_val_lgbm[col].astype(str).fillna('Unknown').astype('category')
                 X_test_lgbm[col] = X_test_lgbm[col].astype(str).fillna('Unknown').astype('category')
                 # Store the learned categories from train
                 category_map[col] = X_train_lgbm[col].dtype
        else:
             print(f"Warning: Categorical feature '{col}' not found in training data columns for LGBM.")

    # Ensure correct feature list for scaling
    active_numerical_features = [f for f in numerical_features if f in X_train_lgbm.columns]

    # Scale Numerical Features
    scaler = None
    features_scaled = []
    if active_numerical_features: # Check if list is not empty
        print("Scaling numerical features for LGBM...")
        scaler = StandardScaler()
        try:
            X_train_lgbm[active_numerical_features] = scaler.fit_transform(X_train_lgbm[active_numerical_features])
            X_val_lgbm[active_numerical_features] = scaler.transform(X_val_lgbm[active_numerical_features])
            X_test_lgbm[active_numerical_features] = scaler.transform(X_test_lgbm[active_numerical_features])
            features_scaled = active_numerical_features
            print("Numerical scaling complete.")
        except ValueError as e:
            print(f"Warning: ValueError during scaling: {e}. Some numerical columns might have non-numeric data or all NaNs. Skipping scaling.")
            scaler = None # Reset scaler if it failed
            features_scaled = []
        except Exception as e:
             print(f"Warning: An unexpected error occurred during scaling: {e}. Skipping scaling.")
             scaler = None
             features_scaled = []
    else:
        print("No numerical features to scale for LGBM.")


    # Prepare LGBM Datasets
    final_feature_names = active_numerical_features + active_categorical_features
    # Ensure final features exist in the dataframe after potential scaling issues etc.
    final_feature_names = [col for col in final_feature_names if col in X_train_lgbm.columns]
    X_train_lgbm = X_train_lgbm[final_feature_names]
    X_val_lgbm = X_val_lgbm[final_feature_names]
    X_test_lgbm = X_test_lgbm[final_feature_names]

    print("Creating LGBM datasets...")
    lgb_train = lgb.Dataset(X_train_lgbm, label=y_train,
                            categorical_feature=active_categorical_features if active_categorical_features else 'auto',
                            feature_name=final_feature_names,
                            free_raw_data=False)
    lgb_eval = lgb.Dataset(X_val_lgbm, label=y_val, reference=lgb_train,
                           categorical_feature=active_categorical_features if active_categorical_features else 'auto',
                           feature_name=final_feature_names,
                           free_raw_data=False)

    # Class Weights for LGBM -> Sample Weights
    sample_weight = None
    if class_weight_dict:
        print("Calculating sample weights for LGBM...")
        try:
             # Ensure y_train indices match weights
             sample_weight = y_train.map(class_weight_dict).fillna(1.0).values
             print(f"Sample weights calculated (min: {np.min(sample_weight):.2f}, max: {np.max(sample_weight):.2f}).")
             lgb_train.set_weight(sample_weight)
             print("Applied sample weights to training dataset.")
        except Exception as e:
             print(f"Warning: Could not compute/apply sample weights for LGBM: {e}.")
             sample_weight = None

    # Define LGBM Parameters (Binary)
    params = {
        'objective': 'binary', 'metric': ['binary_logloss', 'binary_error', 'auc'],
        'boosting_type': 'gbdt', 'n_estimators': 2500, # Can be increased if needed
        'learning_rate': 0.02, 'num_leaves': 31, # Can be tuned
        'reg_alpha': 0.1, 'reg_lambda': 0.1, # Regularization
        'colsample_bytree': 0.8, 'subsample': 0.8, # Feature/Data sampling
        'min_child_samples': 20, # Regularization
        'max_depth': -1, # No limit by default
        'seed': 42, 'n_jobs': -1, 'verbose': -1,
        # 'is_unbalance': True, # Alternative/addition to sample_weight - experiment if needed
    }

    # Train LightGBM Model
    print("Starting LGBM model training...")
    evals_result = {}
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=True), # Stop if val score doesn't improve
        lgb.log_evaluation(period=10), # Print results every 50 rounds
        lgb.record_evaluation(evals_result) # Store results for plotting/analysis if needed
    ]
    lgbm_model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_eval],
                           valid_names=['train', 'eval'], callbacks=callbacks)
    print("LGBM Training finished.")

    # Evaluate LightGBM Model
    print("\nEvaluating LGBM model on the test set...")
    try:
        y_pred_proba = lgbm_model.predict(X_test_lgbm, num_iteration=lgbm_model.best_iteration)
        y_pred_binary = (y_pred_proba > 0.45).astype(int) # Using default 0.5 threshold
        accuracy = accuracy_score(y_test, y_pred_binary)
        auc = roc_auc_score(y_test, y_pred_proba)
        print(f"LGBM Test Accuracy: {accuracy:.4f}")
        print(f"LGBM Test AUC: {auc:.4f}")
        print("\nLGBM Classification Report (Test Set):")
        print(classification_report(y_test, y_pred_binary, target_names=['Move (0)', 'Switch (1)']))
    except Exception as e:
        print(f"Error during LGBM evaluation: {e}")


    # Save Model and Feature Info
    model_save_path = f'switch_predictor_lgbm_model_v2_{model_suffix}.txt' # Adjusted name version
    print(f"Saving LGBM model to {model_save_path}")
    try:
        lgbm_model.save_model(model_save_path)
        print("LGBM Model saved.")
    except Exception as e:
        print(f"Error saving LGBM model: {e}")

    lgbm_info_path = f'switch_predictor_lgbm_feature_info_v2_{model_suffix}.joblib' # Adjusted name version
    print(f"Saving LGBM feature info to {lgbm_info_path}")
    try:
        # Use active_numerical_features and active_categorical_features which reflect the actual columns used
        lgbm_info = {
            'numerical_features': active_numerical_features,
            'categorical_features': active_categorical_features,
            'feature_names_in_order': final_feature_names,
            'features_scaled': features_scaled, # Record which numerical features were scaled
            'category_map': category_map # Store category mappings used
        }
        joblib.dump(lgbm_info, lgbm_info_path)
        print(f"LGBM feature info saved.")
    except Exception as e:
         print(f"Error saving LGBM feature info: {e}")

    # Save scaler only if it was successfully created and used
    if scaler and features_scaled:
        scaler_path = f'switch_predictor_lgbm_scaler_v2_{model_suffix}.joblib' # Adjusted name version
        print(f"Saving LGBM scaler to {scaler_path}")
        try:
             joblib.dump(scaler, scaler_path)
             print(f"LGBM scaler saved.")
        except Exception as e:
             print(f"Error saving LGBM scaler: {e}")

    return lgbm_model


# --- Main execution function (MODIFIED for Simplified + Revealed Moves) ---
def run_switch_training(parquet_path, model_type='tensorflow', feature_set='full',
                       min_turn=0, test_split_size=0.2, val_split_size=0.15,
                       epochs=30, batch_size=256, learning_rate=0.001):
    """Loads data, splits, preprocesses based on feature_set, and trains switch predictor."""

    print(f"--- Starting Switch Predictor Training (V2 - Binary, Redefined Simplified Set) ---")
    print(f"Model type: {model_type.upper()}")
    print(f"Feature Set: {feature_set.upper()}")
    print(f"Loading data from: {parquet_path}")
    try:
        df = pd.read_parquet(parquet_path)
        print(f"Original data shape: {df.shape}")
    except Exception as e:
        print(f"Error loading Parquet file: {e}"); return

    # --- Filter Data ---
    print("\nFiltering data...")
    original_rows = len(df)
    df = df.dropna(subset=['action_taken'])
    print(f"Rows after dropping NaN action_taken: {len(df)}")

    # Apply player filter *conditionally* IF using simplified, maintain original logic
    
    if feature_set == 'simplified' or feature_set == 'minimal_active_species':
         print(f"Filtering for player_to_move == 'p1' (for {feature_set} set)...")
         rows_before_p1_filter = len(df)
         df = df[df['player_to_move'] == 'p1'].copy()
         print(f"Rows after filtering for p1's move: {len(df)} (Removed {rows_before_p1_filter - len(df)})")
         if df.empty: print(f"Error: No data found for player p1's moves (feature_set={feature_set})."); return
    elif feature_set == 'full':
         print("Keeping data for both players ('full' feature set).")

    if min_turn > 0:
        initial_rows_turn_filter = len(df)
        df = df[df['turn_number'] >= min_turn].copy()
        print(f"Rows after filtering turns >= {min_turn}: {len(df)} (Removed {initial_rows_turn_filter - len(df)})")
        if df.empty: print("Error: No data remaining after turn filtering."); return

    # --- Create BINARY Target Variable ---
    print("\nCreating binary target variable 'is_switch' (1 if action starts with 'switch', 0 otherwise)...")
    try:
        df['action_taken'] = df['action_taken'].astype(str)
        y = df['action_taken'].str.startswith('switch').astype(int)
        print(f"Target distribution: 0 (Move) = {(y == 0).sum()}, 1 (Switch) = {(y == 1).sum()}")
        if y.nunique() < 2:
             print("Error: Only one action type (move or switch) found after filtering. Cannot train binary classifier.")
             return
    except KeyError:
        print("Error: 'action_taken' column not found. Cannot create target variable.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during target variable creation: {e}")
        return


    # --- Conditional Feature Selection ---
    X = None
    numerical_features = []
    categorical_features = []
    model_suffix = f"{feature_set}_moves" if feature_set == 'simplified' else feature_set # Add suffix for simplified
    all_revealed_moves_binary_cols = [] # Store names of created move cols

    # --- Feature Set Logic ---
    if feature_set == 'minimal_active_species':
            print("\n--- Using MINIMAL ACTIVE SPECIES feature set ---")
            # --- Extract Active Pokemon SPECIES ONLY ---
            print("  Extracting active Pokemon species...")
            active_species_data = []
            for idx, row in df.iterrows(): # Iterate over the filtered df
                p1_active_species = 'Unknown'
                p2_active_species = 'Unknown'
                for i_slot in range(1, 7):
                    if row.get(f'p1_slot{i_slot}_is_active', 0) == 1:
                        p1_active_species = row.get(f'p1_slot{i_slot}_species', 'Unknown')
                        break # Found p1 active
                for i_slot in range(1, 7):
                    if row.get(f'p2_slot{i_slot}_is_active', 0) == 1:
                        p2_active_species = row.get(f'p2_slot{i_slot}_species', 'Unknown')
                        break # Found p2 active
                active_species_data.append({
                    'original_index': idx,
                    'p1_active_species': p1_active_species,
                    'p2_active_species': p2_active_species
                })
            X = pd.DataFrame(active_species_data).set_index('original_index')
            print(f"  Active species extracted. X shape: {X.shape}")
            del active_species_data; gc.collect()

            # --- Identify Final Feature Types for Minimal Set ---
            categorical_features = ['p1_active_species', 'p2_active_species']
            numerical_features = [] # No numerical features in this set
            print("\nIdentifying final feature types for 'minimal_active_species' set...")
            # Ensure these columns actually exist in X, though they should by construction
            categorical_features = [col for col in categorical_features if col in X.columns]

    elif feature_set == 'simplified':
        print("\n--- Using REDEFINED SIMPLIFIED feature set + REVEALED MOVES ---")
        # Define the base columns for the simplified set (EXCLUDING revealed moves for now)
        selected_columns = ['turn_number', 'last_move_p1', 'last_move_p2', 'field_weather']
        for i in range(1, 7):
            for player in ['p1', 'p2']:
                selected_columns.extend([
                    f'{player}_slot{i}_species', f'{player}_slot{i}_hp_perc',
                    f'{player}_slot{i}_status', f'{player}_slot{i}_is_active',
                    f'{player}_slot{i}_is_fainted' # Keep fainted status
                ])
        hazard_types = ['stealthrock', 'spikes', 'toxicspikes', 'stickyweb']
        for player in ['p1', 'p2']:
             for hazard in hazard_types: selected_columns.append(f'{player}_hazard_{hazard}')
        side_cond_types = ['reflect', 'lightscreen', 'auroraveil', 'tailwind']
        for player in ['p1', 'p2']:
            for cond in side_cond_types: selected_columns.append(f'{player}_side_{cond}')

        print(f"Selecting {len(selected_columns)} base columns for simplified set...")
        valid_selected_columns = [col for col in selected_columns if col in df.columns]
        print(f"  Found {len(valid_selected_columns)} direct base columns.")
        missing_base_cols = set(selected_columns) - set(valid_selected_columns)
        if missing_base_cols: print(f"  Warning: Missing expected base columns: {missing_base_cols}")
        X_simplified_base = df[valid_selected_columns].copy()

        # --- Extract Active Pokemon Details ---
        print("  Extracting active Pokemon details...")
        active_data_list = []
        base_active_features = ['species', 'hp_perc', 'status', 'terastallized'] # Add terastallized here
        for idx, row in df.iterrows(): # Iterate over the filtered df
            active_p1_slot, active_p2_slot = -1, -1
            for i_slot in range(1, 7):
                if row.get(f'p1_slot{i_slot}_is_active', 0) == 1: active_p1_slot = i_slot
                if row.get(f'p2_slot{i_slot}_is_active', 0) == 1: active_p2_slot = i_slot
            row_active_data = {'original_index': idx}
            for player, active_slot in [('p1', active_p1_slot), ('p2', active_p2_slot)]:
                if active_slot != -1:
                    for feat in base_active_features: row_active_data[f'{player}_active_{feat}'] = row.get(f'{player}_slot{active_slot}_{feat}', None)
                else:
                    for feat in base_active_features: row_active_data[f'{player}_active_{feat}'] = None
            active_data_list.append(row_active_data)
        active_df = pd.DataFrame(active_data_list).set_index('original_index')
        print("  Active Pokemon details extracted.")

        # --- Process Revealed Moves (Adapted from 'full' set logic) ---
        print("\n  Processing 'revealed_moves' features for simplified set (Multi-Hot Encoding)...")
        # Identify revealed move columns FROM THE ORIGINAL df
        revealed_move_cols = sorted([col for col in df.columns if col.endswith('_revealed_moves')])
        all_revealed_moves = set()
        new_move_cols_data = {}
        all_revealed_moves_binary_cols = [] # Reset list for this feature set

        if not revealed_move_cols:
            print("  Warning: No '*_revealed_moves' columns found in the original DataFrame.")
        else:
            print("  Finding unique revealed moves...")
            # Find unique moves across all relevant columns in the filtered df
            for col in revealed_move_cols:
                if col in df.columns: # Check if column exists in the filtered df
                    unique_in_col = df[col].dropna().astype(str).str.split(',').explode().unique()
                    # Clean moves: remove empty strings, 'none', 'error_state' etc.
                    all_revealed_moves.update(m.strip() for m in unique_in_col if m and m.strip() not in ['none', 'error_state', '', 'nan'])
            unique_moves_list = sorted(list(all_revealed_moves))
            print(f"  Found {len(unique_moves_list)} unique revealed moves across relevant slots.")

            print("  Creating and populating binary revealed move columns...")
            # Use a function for sanitizing to avoid repetition
            def sanitize_name(name):
                 # Keep it simple: replace non-alphanumeric with underscore
                 return re.sub(r'[^a-zA-Z0-9]+', '_', name).lower()

            # Process each revealed_move column present in the filtered df
            for base_col in revealed_move_cols:
                if base_col not in df.columns: continue
                try:
                    # Crucially, use the index from X_simplified_base to align
                    revealed_sets = df.loc[X_simplified_base.index, base_col].fillna('none').astype(str).str.split(',').apply(set)
                    for move in unique_moves_list:
                        sanitized_move_name = sanitize_name(move)
                        new_col_name = f"{base_col}_{sanitized_move_name}"
                        # Create series directly aligned with X_simplified_base index
                        new_move_cols_data[new_col_name] = revealed_sets.apply(lambda move_set: 1 if move in move_set else 0).astype(np.int8)
                        all_revealed_moves_binary_cols.append(new_col_name) # Store the name
                    del revealed_sets
                except Exception as e:
                     print(f"  Error processing revealed moves in column {base_col}: {e}")
                gc.collect()

            if new_move_cols_data:
                 # Create DataFrame from the collected Series
                 move_features_df = pd.DataFrame(new_move_cols_data) # Index should align now
                 print(f"  Created {len(all_revealed_moves_binary_cols)} new binary move features.")
                 # Concatenate base features, active features, and move features
                 print("  Adding active Pokemon and revealed move details to X DataFrame...")
                 X = pd.concat([X_simplified_base, active_df, move_features_df], axis=1)
                 del X_simplified_base, active_df, move_features_df, new_move_cols_data; gc.collect()
                 print(f"Simplified X final shape: {X.shape}")
            else:
                print("  No new move columns created. Using only base and active features.")
                X = pd.concat([X_simplified_base, active_df], axis=1) # Concatenate without moves
                del X_simplified_base, active_df; gc.collect()
                print(f"Simplified X final shape (no moves): {X.shape}")
        # --- End of Revealed Moves Processing ---

        # --- Identify Final Feature Types for REDEFINED Simplified Set + MOVES ---
        print("\nIdentifying final feature types for 'simplified + moves' set...")
        numerical_features = []
        categorical_features = []

        # Base features
        if 'turn_number' in X.columns: numerical_features.append('turn_number')
        if 'last_move_p1' in X.columns: categorical_features.append('last_move_p1')
        if 'last_move_p2' in X.columns: categorical_features.append('last_move_p2')
        if 'field_weather' in X.columns: categorical_features.append('field_weather')

        # Slot features
        for i in range(1, 7):
            for player in ['p1', 'p2']:
                if f'{player}_slot{i}_species' in X.columns: categorical_features.append(f'{player}_slot{i}_species')
                if f'{player}_slot{i}_hp_perc' in X.columns: numerical_features.append(f'{player}_slot{i}_hp_perc')
                if f'{player}_slot{i}_status' in X.columns: categorical_features.append(f'{player}_slot{i}_status')
                if f'{player}_slot{i}_is_active' in X.columns: numerical_features.append(f'{player}_slot{i}_is_active') # Binary numeric
                if f'{player}_slot{i}_is_fainted' in X.columns: numerical_features.append(f'{player}_slot{i}_is_fainted') # Binary numeric


        # Hazards (numeric level/presence)
        for player in ['p1', 'p2']:
             for hazard in hazard_types:
                  col_name = f'{player}_hazard_{hazard}'
                  if col_name in X.columns: numerical_features.append(col_name)

        # Side conditions (numeric turns/presence)
        for player in ['p1', 'p2']:
            for cond in side_cond_types:
                 col_name = f'{player}_side_{cond}'
                 if col_name in X.columns: numerical_features.append(col_name)

        # Active pokemon features
        for player in ['p1', 'p2']:
            if f'{player}_active_species' in X.columns: categorical_features.append(f'{player}_active_species')
            if f'{player}_active_hp_perc' in X.columns: numerical_features.append(f'{player}_active_hp_perc')
            if f'{player}_active_status' in X.columns: categorical_features.append(f'{player}_active_status')
            # Treat terastallized as binary numeric (0/1 or bool)
            if f'{player}_active_terastallized' in X.columns: numerical_features.append(f'{player}_active_terastallized')

        # *** ADD REVEALED MOVES TO NUMERICAL ***
        if all_revealed_moves_binary_cols:
             print(f"  Adding {len(all_revealed_moves_binary_cols)} binary revealed move features to numerical list.")
             numerical_features.extend(all_revealed_moves_binary_cols)

        # Final cleanup of feature lists
        all_simplified_cols = list(X.columns)
        numerical_features = sorted(list(set([f for f in numerical_features if f in all_simplified_cols])))
        categorical_features = sorted(list(set([f for f in categorical_features if f in all_simplified_cols])))
        overlap = set(numerical_features) & set(categorical_features)
        if overlap:
            print(f"Warning: Overlap detected between numerical and categorical: {overlap}. Removing from numerical.")
            numerical_features = [f for f in numerical_features if f not in overlap]

    elif feature_set == 'full':
        # (Keep 'full' logic - unchanged, including its own revealed moves processing)
        print("\n--- Using FULL feature set ---")
        print("\nPreparing features...")
        base_exclude = ['replay_id', 'action_taken', 'battle_winner', 'player_to_move'] # Add player_to_move if needed
        cols_to_exclude = base_exclude
        feature_columns = [col for col in df.columns if col not in cols_to_exclude]
        X = df[feature_columns].copy()
        print(f"Initial feature count: {len(feature_columns)}")

        # --- Multi-Hot Encode ALL Revealed Moves (Full Set Only) ---
        print("\nProcessing 'revealed_moves' features (Multi-Hot Encoding - ALL SLOTS)...")
        revealed_move_cols = sorted([col for col in X.columns if col.endswith('_revealed_moves')])
        all_revealed_moves = set()
        new_binary_move_cols = [] # Specific list for 'full'

        if not revealed_move_cols:
            print("Warning: No '*_revealed_moves' columns found.")
        else:
            print("  Finding unique revealed moves...")
            for col in revealed_move_cols:
                unique_in_col = X[col].dropna().astype(str).str.split(',').explode().unique()
                # Clean moves
                all_revealed_moves.update(m.strip() for m in unique_in_col if m and m.strip() not in ['none', 'error_state', '', 'nan'])
            unique_moves_list = sorted(list(all_revealed_moves))
            print(f"  Found {len(unique_moves_list)} unique revealed moves across all slots.")

            print("  Creating and populating binary revealed move columns...")
            new_move_cols_data = {}
            def sanitize_name(name): # Use same sanitize function
                 return re.sub(r'[^a-zA-Z0-9]+', '_', name).lower()

            for base_col in revealed_move_cols:
                if base_col not in X.columns: continue
                try:
                    revealed_sets = X[base_col].fillna('none').astype(str).str.split(',').apply(set)
                    for move in unique_moves_list:
                        sanitized_move_name = sanitize_name(move)
                        new_col_name = f"{base_col}_{sanitized_move_name}"
                        new_move_cols_data[new_col_name] = revealed_sets.apply(lambda move_set: 1 if move in move_set else 0).astype(np.int8)
                        new_binary_move_cols.append(new_col_name) # Use the 'full' specific list
                    del revealed_sets
                except Exception as e:
                     print(f"  Error processing revealed moves in column {base_col}: {e}")
                gc.collect()

            if new_move_cols_data:
                 X = pd.concat([X, pd.DataFrame(new_move_cols_data, index=X.index)], axis=1)
                 print(f"  Created {len(new_binary_move_cols)} new binary move features.")
                 all_revealed_moves_binary_cols = new_binary_move_cols # Assign to the general var if needed later? Or keep separate.
            else: print("  No new move columns created.")

            print("  Dropping original revealed_moves string columns...")
            # Ensure columns exist before dropping
            cols_to_drop = [col for col in revealed_move_cols if col in X.columns]
            if cols_to_drop:
                X = X.drop(columns=cols_to_drop)
            gc.collect()

        # --- Identify Feature Types and Handle NaNs (Full Set Specifics) ---
        print("\nIdentifying final feature types and handling remaining NaNs for 'full' set...")
        # Convert object columns to category AFTER filling NaNs
        obj_cols = X.select_dtypes(include=['object']).columns.tolist()
        for col in obj_cols:
             # Fill NaN first, then convert
             X[col] = X[col].fillna('Unknown')
             try: X[col] = X[col].astype('category')
             except TypeError: print(f"  Warning: Could not convert column '{col}' to category.")

        # Handle boolean columns
        bool_cols = X.select_dtypes(include=['boolean', 'bool']).columns.tolist()
        for col in bool_cols: X[col] = X[col].fillna(False).astype('int8') # Treat bools as numeric 0/1

        # Identify numerical and categorical features
        numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        # Explicitly add binary move cols created for 'full' to numerical
        numerical_features.extend(new_binary_move_cols) # Use the 'full' list
        numerical_features = sorted(list(set(numerical_features)))

        categorical_features = sorted(list(set(X.select_dtypes(include=['category']).columns.tolist())))

        # Resolve overlaps
        overlap = set(numerical_features) & set(categorical_features)
        if overlap:
            print(f"Warning: Overlap detected between numerical and categorical: {overlap}. Removing from numerical.")
            numerical_features = [f for f in numerical_features if f not in overlap]

        # Final NaN fill for numerical (should ideally happen after identifying types)
        if numerical_features:
             # Select only numerical columns that are actually in X
             num_cols_in_X = [col for col in numerical_features if col in X.columns]
             if num_cols_in_X:
                 nan_counts = X[num_cols_in_X].isnull().sum()
                 cols_with_nan = nan_counts[nan_counts > 0].index.tolist()
                 if cols_with_nan:
                     print(f"  Numerical columns still have NaNs: {cols_with_nan}. Filling with 0.")
                     X[num_cols_in_X] = X[num_cols_in_X].fillna(0)
                 # Ensure they are numeric after filling
                 for col in num_cols_in_X:
                     X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0) # Coerce ensures numeric type

    else:
        # This case should not be reached if argparse choices are updated
        print(f"Error: Invalid feature_set '{feature_set}'. Choose 'full' or 'simplified'.")
        return

    # --- Fill NaNs (General Check - applied AFTER feature set specific handling) ---
    # This section might be less necessary if NaNs are handled well within each feature set block,
    # but can serve as a final safety net.
    print(f"\nFinal NaN Check for '{feature_set}' set...")
    nan_report_before = X.isnull().sum()
    cols_with_nan_before = nan_report_before[nan_report_before > 0]
    if not cols_with_nan_before.empty:
        print(f"  NaNs found BEFORE final handling in columns: {cols_with_nan_before.index.tolist()}")
        # Re-identify types based on the final state of X
        final_numerical = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
        final_categorical = [col for col in X.columns if pd.api.types.is_categorical_dtype(X[col]) or X[col].dtype == 'object']

        print("  Applying final fillna round...")
        for col in final_numerical:
            if X[col].isnull().any():
                 # Check if filling with 0 is appropriate (might not be for all features)
                 print(f"    Filling NaNs in numerical column '{col}' with 0.")
                 X[col] = X[col].fillna(0)

        default_cat_fill = 'Unknown'
        for col in final_categorical:
             if X[col].isnull().any():
                  print(f"    Filling NaNs in categorical/object column '{col}' with '{default_cat_fill}'.")
                  # Ensure category exists before filling if it's already categorical
                  if pd.api.types.is_categorical_dtype(X[col]):
                      if default_cat_fill not in X[col].cat.categories:
                           try:
                               # Add category in-place
                               X[col].cat.add_categories([default_cat_fill], inplace=True)
                           except Exception as e:
                               print(f"      Warning: Could not add category '{default_cat_fill}' to {col}. Error: {e}")
                      X[col] = X[col].fillna(default_cat_fill)
                  else:
                      # Convert object cols to category after filling
                       X[col] = X[col].fillna(default_cat_fill).astype('category')

        print("  Final NaN handling applied.")
    else:
        print("  No NaNs found before final encoding/scaling.")

    # Final Check - Crucial before splitting
    nan_report_after = X.isnull().sum()
    if nan_report_after.sum() > 0:
        print("Error: NaNs still present AFTER final handling. Columns:")
        print(nan_report_after[nan_report_after > 0])
        print("Cannot proceed with training. Check data types and fill logic.")
        # You might want to inspect X[nan_report_after[nan_report_after > 0].index].info() here
        return


    # --- Common Steps from here ---
    print(f"\nFinal feature counts for '{feature_set}' set:")
    # Ensure lists are based on columns actually present in X
    numerical_features = [f for f in numerical_features if f in X.columns]
    categorical_features = [f for f in categorical_features if f in X.columns]
    print(f"  Numerical: {len(numerical_features)}")
    print(f"  Categorical: {len(categorical_features)}")
    print(f"  Total Features in X: {X.shape[1]}")

    # Verify X columns match the union of feature lists
    expected_cols = set(numerical_features) | set(categorical_features)
    actual_cols = set(X.columns)
    if expected_cols != actual_cols:
        print(f"Warning: Mismatch between identified features and columns in X.")
        print(f"  Missing from X: {expected_cols - actual_cols}")
        print(f"  Extra in X: {actual_cols - expected_cols}")
        # It might be safer to redefine feature lists based on X.columns and dtypes here
        numerical_features = sorted([col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])])
        categorical_features = sorted([col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]) # Assume non-numeric are categorical for simplicity here
        print(f"  Corrected counts - Numerical: {len(numerical_features)}, Categorical: {len(categorical_features)}")


    if not numerical_features and not categorical_features:
         print("Error: No features identified for training.")
         return
    if X.empty:
        print("Error: Feature DataFrame X is empty before splitting.")
        return

    # --- Split Data ---
    print("\nSplitting data into Train, Validation, Test sets...")
    try:
        # Check if stratification is possible
        if y.nunique() < 2:
            print("Warning: Only one class in target 'y'. Cannot stratify. Splitting randomly.")
            stratify_param = None
        elif y.value_counts().min() < 2: # Need at least 2 samples of the smallest class for stratification
             print(f"Warning: Smallest class count ({y.value_counts().min()}) is less than 2. Cannot stratify. Splitting randomly.")
             stratify_param = None
        else:
             stratify_param = y

        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=test_split_size, random_state=42, stratify=stratify_param
        )

        # Relative validation size calculation
        train_full_size = 1.0 - test_split_size
        val_size_relative = val_split_size / train_full_size if train_full_size > 0 else 0.15 # Default if calculation fails
        if not (0 < val_size_relative < 1):
            print(f"Warning: Calculated relative validation size ({val_size_relative:.2f}) is invalid. Using default 0.15.")
            val_size_relative = 0.15

        # Check stratification for second split
        if y_train_full.nunique() < 2:
             stratify_param_2 = None
        elif y_train_full.value_counts().min() < 2:
             stratify_param_2 = None
        else:
             stratify_param_2 = y_train_full

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=val_size_relative, random_state=42, stratify=stratify_param_2
        )
        print("Successfully split data.") # Removed 'with stratification' as it might not always happen

    except Exception as e: # Catch broader exceptions during split
        print(f"Error during data splitting: {e}. Cannot proceed.")
        return

    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")
    print(f"Train target distribution: 0={ (y_train == 0).sum() }, 1={ (y_train == 1).sum() }")
    print(f"Val target distribution:   0={ (y_val == 0).sum() }, 1={ (y_val == 1).sum() }")
    print(f"Test target distribution:  0={ (y_test == 0).sum() }, 1={ (y_test == 1).sum() }")

    # --- Explicit Cleanup ---
    del X, y, df, X_train_full, y_train_full
    if 'X_simplified_base' in locals(): del X_simplified_base
    if 'active_df' in locals(): del active_df
    if 'move_features_df' in locals(): del move_features_df
    gc.collect()


    # --- Calculate Class Weights ---
    print("\nCalculating class weights for handling imbalance...")
    class_weight_dict = None
    try:
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        if len(unique_classes) == 2:
             # Use sklearn's utility
             class_weights_values = compute_class_weight('balanced', classes=unique_classes, y=y_train)
             # Map numpy types to standard Python types if needed, ensure correct keys (0 and 1)
             class_weight_dict = {int(cls): float(weight) for cls, weight in zip(unique_classes, class_weights_values)}
             # Ensure both 0 and 1 keys exist, even if one class was missing in train (shouldn't happen with split logic)
             class_weight_dict.setdefault(0, 1.0)
             class_weight_dict.setdefault(1, 1.0)
             print(f"Class weights calculated: {class_weight_dict}")
        else:
            print("Warning: Only one class present in y_train after splitting. Using uniform weights (1.0).")
            class_weight_dict = {0: 1.0, 1: 1.0} # Default for safety
    except Exception as e:
         print(f"Error calculating class weights: {e}. Using uniform weights (1.0).")
         class_weight_dict = {0: 1.0, 1: 1.0}


    # --- Preprocess X data based on model type ---
    X_train_processed, X_val_processed, X_test_processed = None, None, None
    preprocessor = None
    # Define paths for saving preprocessing artifacts (using v2 in names)
    feature_lists_path = f'switch_predictor_feature_lists_v2_{model_suffix}.joblib'
    preprocessor_path = f'switch_predictor_tf_preprocessor_v2_{model_suffix}.joblib'

    # Final check on feature lists based on X_train columns just before preprocessing
    final_train_cols = X_train.columns.tolist()
    numerical_features = [f for f in numerical_features if f in final_train_cols]
    categorical_features = [f for f in categorical_features if f in final_train_cols]

    # Save final feature lists used for preprocessing/training
    try:
         joblib.dump({
             'feature_columns_final': final_train_cols,
             'numerical_features': numerical_features,
             'categorical_features': categorical_features
             }, feature_lists_path)
         print(f"Final feature lists saved to {feature_lists_path}")
    except Exception as e: print(f"Error saving feature lists: {e}")

    if model_type == 'tensorflow':
        print("\nSetting up TF preprocessing pipeline (OneHotEncoder + Scaler)...")
        transformers = []
        # Ensure lists contain valid columns from X_train
        tf_numerical_features = [f for f in numerical_features if f in X_train.columns]
        tf_categorical_features = [f for f in categorical_features if f in X_train.columns]

        if tf_numerical_features:
            numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
            transformers.append(('num', numerical_transformer, tf_numerical_features))
            print(f"  Added StandardScaler for {len(tf_numerical_features)} numerical features.")
        else: print("  Info: No numerical features to scale for TF.")

        if tf_categorical_features:
            # Use sparse_output=False if dense array is needed or memory allows
            categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))])
            transformers.append(('cat', categorical_transformer, tf_categorical_features))
            print(f"  Added OneHotEncoder for {len(tf_categorical_features)} categorical features.")
        else: print("  Info: No categorical features to OneHotEncode for TF.")

        if not transformers: print("Error: No features to process for TF!"); return

        # Use sparse_threshold=0.3 to keep sparse if density < 30%
        preprocessor = ColumnTransformer(transformers=transformers, remainder='drop', sparse_threshold=0.3)

        print("Applying TF preprocessing (fit on train, transform all)...")
        try:
            # Fit only on TRAIN data
            X_train_processed = preprocessor.fit_transform(X_train)
            print(f"  Fit TF preprocessor on training data (Output shape: {X_train_processed.shape}, Type: {type(X_train_processed)})")
            # Transform VAL and TEST data
            X_val_processed = preprocessor.transform(X_val)
            X_test_processed = preprocessor.transform(X_test)
            print(f"TF Processed shapes - Train: {X_train_processed.shape}, Val: {X_val_processed.shape}, Test: {X_test_processed.shape}")

            # Save the FITTED preprocessor
            joblib.dump(preprocessor, preprocessor_path)
            print(f"TF preprocessor saved to {preprocessor_path}")
        except ValueError as e:
            print(f"ValueError during TF preprocessing: {e}")
            print("Check for non-numeric data in numerical columns or unexpected values in categorical columns.")
            # Optionally print info for columns causing issues if identifiable
            return
        except MemoryError: print("MemoryError during TF preprocessing. Try reducing features or using sparse matrices if possible."); return
        except Exception as e:
             print(f"An unexpected error during TF preprocessing: {e}")
             import traceback; traceback.print_exc()
             return
        del X_train, X_val, X_test; gc.collect() # Clean up raw data after processing

    elif model_type == 'lightgbm':
        print("\nPreprocessing for LGBM (dtype conversion, scaling) will occur inside its training function.")
        # Pass the raw (but split) DataFrames to the LGBM function
        X_train_processed, X_val_processed, X_test_processed = X_train, X_val, X_test
        # Note: We pass the *original* numerical/categorical lists identified earlier.
        # The LGBM function will handle dtype conversion and scaling internally.
    else:
        print(f"Error: Unknown model_type '{model_type}'"); return


    # --- Train Selected Model ---
    print(f"\n--- Initiating {model_type.upper()} Model Training ({model_suffix} features V2, Binary Switch Prediction) ---")
    if model_type == 'tensorflow':
         if X_train_processed is not None:
             train_tensorflow_switch_predictor(X_train_processed, X_val_processed, X_test_processed,
                                               y_train, y_val, y_test,
                                               class_weight_dict,
                                               epochs, batch_size, learning_rate,
                                               model_suffix=model_suffix) # Pass the potentially updated suffix
         else: print("Skipping TF training due to preprocessing errors.")
    elif model_type == 'lightgbm':
         # Pass the original data splits and feature lists
         train_lgbm_switch_predictor(X_train_processed, X_val_processed, X_test_processed,
                                     y_train, y_val, y_test,
                                     numerical_features, # Pass the final list based on X_train
                                     categorical_features, # Pass the final list based on X_train
                                     class_weight_dict,
                                     model_suffix=model_suffix) # Pass the potentially updated suffix
    print("\n--- Switch Predictor Training Script Finished ---")


# --- Main execution block (unchanged argparse) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a binary switch predictor (V2 - Switch vs Move).")
    parser.add_argument("parquet_file", type=str, help="Path to the input Parquet file.")
    parser.add_argument("--model_type", choices=['tensorflow', 'lightgbm'], default='lightgbm', help="Type of model to train.")
    parser.add_argument("--feature_set", choices=['full', 'simplified','minimal_active_species'], default='simplified', help="Feature set to use.")
    parser.add_argument("--min_turn", type=int, default=1, help="Minimum turn number to include (default: 1).")
    parser.add_argument("--test_split", type=float, default=0.2, help="Fraction for test set (default: 0.2).")
    parser.add_argument("--val_split", type=float, default=0.15, help="Fraction for validation set (relative to train data, default: 0.15).")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs (TF only).")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size (TF only).")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (TF only).")

    args = parser.parse_args()

    # Validate split sizes
    if not (0 < args.test_split < 1): print("Error: test_split must be between 0 and 1."); exit(1)
    if not (0 < args.val_split < 1): print("Error: val_split must be between 0 and 1."); exit(1)

    # Call the main training function
    run_switch_training(
        parquet_path=args.parquet_file,
        model_type=args.model_type,
        feature_set=args.feature_set,
        min_turn=args.min_turn,
        test_split_size=args.test_split,
        val_split_size=args.val_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )