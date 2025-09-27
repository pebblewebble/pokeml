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
from tensorflow.keras.layers import BatchNormalization, Activation

# Suppress TensorFlow/warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning) # Ignore pandas 3.0 warnings for now

# --- Helper Function (find_active_species - unchanged) ---
def find_active_species(row, player_prefix):
    """
    Finds the species of the active Pokemon for a given player prefix ('p1' or 'p2')
    in a DataFrame row containing slot information.
    """
    for i in range(1, 7): # Check slots 1 to 6
        active_col = f"{player_prefix}_slot{i}_is_active"
        species_col = f"{player_prefix}_slot{i}_species"
        # Check if both columns exist in the row's index (safer than assuming they do)
        if active_col in row.index and species_col in row.index:
             # Check if the active flag is explicitly 1 (not NaN or other values)
            if row[active_col] == 1:
                # Return the species, handle potential None/NaN from species column itself
                return row[species_col] if pd.notna(row[species_col]) else 'Unknown'
    return 'Unknown' # Return 'Unknown' if no active Pokemon found

# --- TF Training Function (MODIFIED to accept suffix) ---
def train_tensorflow_action_predictor(X_train_processed, X_val_processed, X_test_processed,
                                      y_train_encoded, y_val_encoded, y_test_encoded, # INTEGER encoded y
                                      num_classes, class_weight_dict, label_encoder, # Pass num_classes, weights, encoder
                                      epochs=20, batch_size=128, learning_rate=0.001,
                                      label_suffix=""): # <--- MODIFIED: Added label_suffix
    print(f"\n--- Training TensorFlow Model ---")
    print(f"Input shape: {X_train_processed.shape[1]}")
    print(f"Num classes: {num_classes}")

    # Convert integer labels to one-hot encoding
    print("One-hot encoding target variable for TF...")
    try:
        y_train_one_hot = to_categorical(y_train_encoded, num_classes=num_classes)
        y_val_one_hot = to_categorical(y_val_encoded, num_classes=num_classes)
        y_test_one_hot = to_categorical(y_test_encoded, num_classes=num_classes)
        print("One-hot encoding complete.")
    except ValueError as e:
         print(f"Error during one-hot encoding: {e}")
         return None, None
    except MemoryError:
        print("MemoryError during one-hot encoding.")
        return None, None

    # Define the Model
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
        Dense(num_classes, activation='softmax')
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')])
    model.summary()

    # Train the Model
    print("\nStarting TF model training...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    history = model.fit(
        X_train_processed, y_train_one_hot,
        validation_data=(X_val_processed, y_val_one_hot),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=[early_stopping],
        verbose=2
    )
    print("TF Training finished.")

    # Evaluate the Model
    print("\nEvaluating TF model on the test set...")
    results = model.evaluate(X_test_processed, y_test_one_hot, verbose=0)
    loss = results[0]
    accuracy = results[1]
    top_5_accuracy = results[2] if len(results) > 2 else np.nan
    print(f"TF Test Loss: {loss:.4f}")
    print(f"TF Test Accuracy: {accuracy:.4f}")
    print(f"TF Test Top-5 Accuracy: {top_5_accuracy:.4f}")

    # Save Model
    # <--- MODIFIED: Use label_suffix in filename --->
    model_save_path = f'action_tf_model_v4_{label_suffix}.keras'
    print(f"Saving TF model to {model_save_path}")
    try:
        model.save(model_save_path)
        print("TF Model saved.")
    except Exception as e:
        print(f"Error saving TF model: {e}")

    return history, model


# --- LGBM Training Function (MODIFIED to accept suffix) ---
def train_lgbm_action_predictor(X_train, X_val, X_test,
                                y_train_encoded, y_val_encoded, y_test_encoded, # Use INTEGER encoded y
                                numerical_features, categorical_features,
                                num_classes, class_weight_dict, label_encoder, # Pass num_classes, weights, encoder
                                label_suffix=""): # <--- MODIFIED: Added label_suffix
    print(f"\n--- Training LightGBM Model ---")
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
            all_categories = pd.concat([
                X_train_lgbm[col].astype(str),
                X_val_lgbm[col].astype(str),
                X_test_lgbm[col].astype(str)
            ]).unique()
            cat_type = pd.CategoricalDtype(categories=all_categories, ordered=False)
            X_train_lgbm[col] = X_train_lgbm[col].astype(str).astype(cat_type)
            X_val_lgbm[col] = X_val_lgbm[col].astype(str).astype(cat_type)
            X_test_lgbm[col] = X_test_lgbm[col].astype(str).astype(cat_type)
            category_map[col] = cat_type
        else:
             print(f"Warning: Categorical feature '{col}' not found in training data columns for LGBM.")

    # Scale Numerical Features
    scaler = None
    active_numerical_features = []
    if numerical_features:
        print("Scaling numerical features for LGBM...")
        scaler = StandardScaler()
        active_numerical_features = [f for f in numerical_features if f in X_train_lgbm.columns]
        if active_numerical_features:
            X_train_lgbm[active_numerical_features] = scaler.fit_transform(X_train_lgbm[active_numerical_features])
            X_val_lgbm[active_numerical_features] = scaler.transform(X_val_lgbm[active_numerical_features])
            X_test_lgbm[active_numerical_features] = scaler.transform(X_test_lgbm[active_numerical_features])
            print("Numerical scaling complete.")
        else:
             print("No valid numerical features found to scale.")
             scaler = None
    else:
        print("No numerical features defined to scale.")

    # Prepare LGBM Datasets
    final_feature_names = active_numerical_features + active_categorical_features
    X_train_lgbm = X_train_lgbm[final_feature_names]
    X_val_lgbm = X_val_lgbm[final_feature_names]
    X_test_lgbm = X_test_lgbm[final_feature_names]
    print("Creating LGBM datasets...")
    lgb_train = lgb.Dataset(X_train_lgbm, label=y_train_encoded,
                            categorical_feature=active_categorical_features if active_categorical_features else 'auto',
                            feature_name=final_feature_names,
                            free_raw_data=False)
    lgb_eval = lgb.Dataset(X_val_lgbm, label=y_val_encoded, reference=lgb_train,
                           categorical_feature=active_categorical_features if active_categorical_features else 'auto',
                           feature_name=final_feature_names,
                           free_raw_data=False)

    # Class Weights for LGBM
    sample_weight = None
    if class_weight_dict:
        print("Calculating sample weights for LGBM...")
        try:
             sample_weight = np.array([class_weight_dict.get(cls_idx, 1.0) for cls_idx in y_train_encoded])
             print(f"Sample weights calculated (min: {np.min(sample_weight):.2f}, max: {np.max(sample_weight):.2f}).")
             lgb_train.set_weight(sample_weight)
             print("Applied sample weights to training dataset.")
        except Exception as e:
             print(f"Warning: Could not compute/apply sample weights for LGBM: {e}.")
             sample_weight = None

    # Define LGBM Parameters
    params = {
        'objective': 'multiclass', 'metric': ['multi_logloss', 'multi_error'],
        'num_class': num_classes, 'boosting_type': 'gbdt', 'n_estimators': 2500,
        'learning_rate': 0.02, 'num_leaves': 21, 'reg_alpha': 0.5, 'reg_lambda': 0.5,
        'colsample_bytree': 0.7, 'subsample': 0.7, 'min_child_samples': 50,
        'max_depth': -1, 'seed': 42, 'n_jobs': -1, 'verbose': -1,
    }

    # Train LightGBM Model
    print("Starting LGBM model training...")
    evals_result = {}
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50),
        lgb.record_evaluation(evals_result)
    ]
    lgbm_model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_eval],
                           valid_names=['train', 'eval'], callbacks=callbacks)
    print("LGBM Training finished.")

    # Evaluate LightGBM Model
    print("\nEvaluating LGBM model on the test set...")
    y_pred_proba = lgbm_model.predict(X_test_lgbm, num_iteration=lgbm_model.best_iteration)
    y_pred_indices = np.argmax(y_pred_proba, axis=1)
    accuracy = accuracy_score(y_test_encoded, y_pred_indices)
    try:
        top_5_accuracy = top_k_accuracy_score(y_test_encoded, y_pred_proba, k=5, labels=np.arange(num_classes))
    except ValueError:
        print("Warning: Cannot calculate top-5 accuracy.")
        top_5_accuracy = np.nan
    print(f"LGBM Test Accuracy: {accuracy:.4f}")
    print(f"LGBM Test Top-5 Accuracy: {top_5_accuracy:.4f}")

    # Save Model and Feature Info
    # <--- MODIFIED: Use label_suffix in filenames --->
    model_save_path = f'action_lgbm_model_v4_{label_suffix}.txt'
    print(f"Saving LGBM model to {model_save_path}")
    try:
        lgbm_model.save_model(model_save_path)
        print("LGBM Model saved.")
    except Exception as e:
        print(f"Error saving LGBM model: {e}")

    lgbm_info_path = f'action_lgbm_feature_info_v4_{label_suffix}.joblib'
    print(f"Saving LGBM feature info to {lgbm_info_path}")
    try:
        lgbm_info = {
            'numerical_features': active_numerical_features,
            'categorical_features': active_categorical_features,
            'feature_names_in_order': final_feature_names,
            'category_map': category_map
        }
        joblib.dump(lgbm_info, lgbm_info_path)
        print(f"LGBM feature info saved.")
    except Exception as e:
         print(f"Error saving LGBM feature info: {e}")

    if scaler and active_numerical_features:
        scaler_path = f'action_lgbm_scaler_v4_{label_suffix}.joblib'
        print(f"Saving LGBM scaler to {scaler_path}")
        try:
             joblib.dump(scaler, scaler_path)
             print(f"LGBM scaler saved.")
        except Exception as e:
             print(f"Error saving LGBM scaler: {e}")

    return lgbm_model


# --- MODIFIED Main execution function ---
def run_action_training(parquet_path, model_type='tensorflow', feature_set='full',
                        predict_mode='move_only', # <--- NEW argument with default
                        min_turn=0, test_split_size=0.2, val_split_size=0.15,
                        epochs=30, batch_size=256, learning_rate=0.001): # <--- MODIFIED signature
    """Loads data, splits, preprocesses based on feature_set, and trains action predictor."""

    print(f"--- Starting Action Predictor Training (V4 - with predict_mode) ---") # <--- MODIFIED Print
    print(f"Model type: {model_type.upper()}")
    print(f"Feature Set: {feature_set.upper()}")
    print(f"Prediction Mode: {predict_mode.upper()}") # <--- NEW Print
    print(f"Loading data from: {parquet_path}")
    try:
        df = pd.read_parquet(parquet_path)
        print(f"Original data shape: {df.shape}")
    except Exception as e:
        print(f"Error loading Parquet file: {e}"); return

    # --- Filter Data ---
    print("\nFiltering data...")
    original_rows = len(df)
    df = df.dropna(subset=['action_taken']) # Essential target
    print(f"Rows after dropping NaN action_taken: {len(df)}")

    # Filter based on player *before* potentially removing player_to_move column
    if feature_set in ['simplified', 'medium']:
        print(f"Filtering for player_to_move == 'p1' (for {feature_set} set)...")
        rows_before_p1_filter = len(df)
        df = df[df['player_to_move'] == 'p1'].copy()
        print(f"Rows after filtering for p1's move: {len(df)} (Removed {rows_before_p1_filter - len(df)})")
        if df.empty: print(f"Error: No data found for player p1's moves (feature_set={feature_set})."); return
    else: # full set uses player_to_move as a feature potentially
        print("Keeping data for both players ('full' feature set).")

    df = df[df['action_taken'].str.startswith('move')].copy()

    # <--- NEW: Filter based on predict_mode --->
    y_raw = None # Initialize
    if predict_mode == 'move_only':
        print(f"\nFiltering for actions starting with 'move:' (predict_mode='move_only')...")
        rows_before_move_filter = len(df)
        # Ensure action_taken is string type before filtering
        df = df[df['action_taken'].astype(str).str.startswith('move:')].copy()
        if df.empty:
            print("Error: No 'move:' actions found after filtering. Cannot train in 'move_only' mode.")
            return
        print(f"Rows after filtering for move actions: {len(df)} (Removed {rows_before_move_filter - len(df)})")
        # Extract only the move name as the target
        y_raw = df['action_taken'].str.replace('move:', '', regex=False)
        print("Target variable 'y_raw' now contains only move names.")
    elif predict_mode == 'all_actions':
        print("\nUsing all actions (moves and switches) as target (predict_mode='all_actions').")
        y_raw = df['action_taken'].copy()
    else:
        print(f"Error: Invalid predict_mode '{predict_mode}'. Choose 'move_only' or 'all_actions'.")
        return
    # <--- End of predict_mode filtering --->


    # Filter based on turn number (applied to potentially filtered df)
    if min_turn > 0:
        initial_rows_turn_filter = len(df)
        df = df[df['turn_number'] >= min_turn].copy()
        print(f"Rows after filtering turns >= {min_turn}: {len(df)} (Removed {initial_rows_turn_filter - len(df)})")
        if df.empty: print("Error: No data remaining after turn filtering."); return

    # --- Conditional Feature Selection and Target Prep ---
    X = None
    # y_raw defined above based on predict_mode
    numerical_features = []
    categorical_features = []
    selected_columns = []
    # <--- MODIFIED: Create a unique suffix for saved files --->
    label_encoder_suffix = f"{feature_set}_{predict_mode}"

    # --- Feature Set Logic (simplified, medium, full) ---
    # NOTE: The 'medium' block below is the one modified in the previous step
    #       to include active revealed moves. Keep that version.
    #       The 'full' and 'simplified' blocks remain as they were in V4.

    if feature_set == 'simplified':
        # (Keep V4 'simplified' logic - only p1_active_species, p2_active_species)
        print("\n--- Using SIMPLIFIED feature set (active species only) ---")
        print("Extracting active species for P1 and P2...")
        p1_active_series = df.apply(lambda row: find_active_species(row, 'p1'), axis=1)
        p2_active_series = df.apply(lambda row: find_active_species(row, 'p2'), axis=1)
        X = pd.DataFrame({
            'p1_active_species': p1_active_series,
            'p2_active_species': p2_active_series
        }, index=df.index)
        print(f"Simplified X shape: {X.shape}")
        numerical_features = []
        categorical_features = ['p1_active_species', 'p2_active_species']


    elif feature_set == 'medium':
        print("\n--- Using MEDIUM feature set (with Active Revealed Moves) ---")
        selected_columns = []
        base_active_features = ['species', 'hp_perc', 'status', 'boost_atk', 'boost_def', 'boost_spa', 'boost_spd', 'boost_spe', 'terastallized', 'tera_type']
        print("Identifying column patterns for non-move features...")
        bench_cols = []
        for i in range(1, 7):
            for player in ['p1', 'p2']:
                 bench_cols.append(f'{player}_slot{i}_hp_perc')
                 bench_cols.append(f'{player}_slot{i}_status')
                 bench_cols.append(f'{player}_slot{i}_species')
        selected_columns.extend(bench_cols)
        field_cols = ['field_weather', 'field_terrain', 'field_pseudo_weather']
        selected_columns.extend(field_cols)
        hazard_cols = []
        hazard_types = ['stealth_rock', 'spikes', 'toxic_spikes', 'sticky_web']
        for player in ['p1', 'p2']:
             for hazard in hazard_types:
                  col_name = f'{player}_hazard_{hazard.replace(" ", "_")}'
                  hazard_cols.append(col_name)
        selected_columns.extend(hazard_cols)
        side_cond_cols = []
        side_cond_types = ['reflect', 'light_screen', 'aurora_veil', 'tailwind']
        for player in ['p1', 'p2']:
            for cond in side_cond_types:
                 col_name = f'{player}_side_{cond.lower().replace(" ", "_")}'
                 side_cond_cols.append(col_name)
        selected_columns.extend(side_cond_cols)
        context_cols = ['last_move_p1', 'last_move_p2', 'turn_number']
        selected_columns.extend(context_cols)

        print(f"Selecting {len(selected_columns)} base columns + preparing for active Pokemon info...")
        valid_selected_columns = [col for col in selected_columns if col in df.columns]
        print(f"  Found {len(valid_selected_columns)} direct columns.")
        missing_base_cols = set(selected_columns) - set(valid_selected_columns)
        if missing_base_cols: print(f"  Warning: Missing base columns: {missing_base_cols}")
        X_medium = df[valid_selected_columns].copy()

        print("  Extracting active Pokemon details (including revealed moves string)...")
        active_data = {}
        source_revealed_move_cols = [col for col in df.columns if col.endswith('_revealed_moves')]
        for i_row, (idx, row) in enumerate(df.iterrows()):
            # ( ... Same active data extraction loop as provided in previous answer ... )
            # --- Start of loop ---
            if i_row > 0 and i_row % 100000 == 0: print(f"    Processed {i_row} rows for active details...")
            active_p1_slot = -1
            active_p2_slot = -1
            # Find active slots for this row
            for i_slot in range(1, 7):
                if row.get(f'p1_slot{i_slot}_is_active', 0) == 1: active_p1_slot = i_slot
                if row.get(f'p2_slot{i_slot}_is_active', 0) == 1: active_p2_slot = i_slot
            row_active_data = {}
            # P1
            p1_active_moves_str = 'none'
            if active_p1_slot != -1:
                for feat in base_active_features:
                     row_active_data[f'p1_active_{feat}'] = row.get(f'p1_slot{active_p1_slot}_{feat}', None)
                p1_moves_col_name = f'p1_slot{active_p1_slot}_revealed_moves'
                if p1_moves_col_name in row.index:
                    p1_active_moves_str = row.get(p1_moves_col_name, 'none')
            else:
                 for feat in base_active_features: row_active_data[f'p1_active_{feat}'] = None
            row_active_data['p1_active_revealed_moves_str'] = p1_active_moves_str
            # P2
            p2_active_moves_str = 'none'
            if active_p2_slot != -1:
                 for feat in base_active_features:
                      row_active_data[f'p2_active_{feat}'] = row.get(f'p2_slot{active_p2_slot}_{feat}', None)
                 p2_moves_col_name = f'p2_slot{active_p2_slot}_revealed_moves'
                 if p2_moves_col_name in row.index:
                    p2_active_moves_str = row.get(p2_moves_col_name, 'none')
            else:
                 for feat in base_active_features: row_active_data[f'p2_active_{feat}'] = None
            row_active_data['p2_active_revealed_moves_str'] = p2_active_moves_str
            active_data[idx] = row_active_data
            # --- End of loop ---
        print("  Active Pokemon details (incl. moves strings) extracted.")

        print("  Adding active Pokemon details to X DataFrame...")
        active_df = pd.DataFrame.from_dict(active_data, orient='index')
        active_df['p1_active_revealed_moves_str'] = active_df['p1_active_revealed_moves_str'].fillna('none').astype(str)
        active_df['p2_active_revealed_moves_str'] = active_df['p2_active_revealed_moves_str'].fillna('none').astype(str)
        X = pd.concat([X_medium, active_df], axis=1)
        print(f"Medium X shape before move encoding: {X.shape}")
        del X_medium, active_df, active_data; gc.collect()

        print("\nProcessing ACTIVE 'revealed_moves' features (Multi-Hot Encoding)...")
        active_revealed_move_cols = ['p1_active_revealed_moves_str', 'p2_active_revealed_moves_str']
        active_all_revealed_moves = set()
        new_binary_move_cols = []
        print("  Finding unique revealed moves from ACTIVE slots...")
        for col in active_revealed_move_cols:
            if col in X.columns:
                unique_in_col = X[col].fillna('none').astype(str).str.split(',').explode().unique()
                active_all_revealed_moves.update(m for m in unique_in_col if m and m != 'none' and m != 'error_state')
            else: print(f"  Warning: Expected active moves column '{col}' not found.")
        if not active_all_revealed_moves:
             print("  Warning: No valid revealed moves found in active slots.")
        else:
            unique_moves_list = sorted(list(active_all_revealed_moves))
            print(f"  Found {len(unique_moves_list)} unique revealed moves across ACTIVE slots.")
            print("  Creating and populating binary revealed move columns for active slots...")
            for base_col in active_revealed_move_cols:
                if base_col not in X.columns: continue
                player_prefix = base_col.split('_')[0]
                new_col_prefix = f"{player_prefix}_active_revealed_move"
                revealed_sets = X[base_col].fillna('none').astype(str).str.split(',').apply(set)
                for move in unique_moves_list:
                    sanitized_move_name = move.replace(' ', '_').replace('-', '_').replace(':', '').replace('%', 'perc')
                    new_col_name = f"{new_col_prefix}_{sanitized_move_name}"
                    X[new_col_name] = revealed_sets.apply(lambda move_set: 1 if move in move_set else 0).astype(np.int8)
                    new_binary_move_cols.append(new_col_name)
                del revealed_sets; gc.collect()
            print(f"  Created {len(new_binary_move_cols)} new binary active move features.")
        print("  Dropping temporary active revealed_moves string columns...")
        cols_to_drop = [col for col in active_revealed_move_cols if col in X.columns]
        if cols_to_drop: X = X.drop(columns=cols_to_drop)
        gc.collect()
        print(f"Medium X shape after move encoding: {X.shape}")

        print("\nIdentifying final feature types for 'medium' set...")
        numerical_features = []
        categorical_features = []
        # ( ... Same final feature type identification as provided in previous answer ... )
        # --- Start feature types ---
        for player in ['p1', 'p2']:
            if f'{player}_active_species' in X.columns: categorical_features.append(f'{player}_active_species')
            if f'{player}_active_status' in X.columns: categorical_features.append(f'{player}_active_status')
            if f'{player}_active_tera_type' in X.columns: categorical_features.append(f'{player}_active_tera_type')
            if f'{player}_active_hp_perc' in X.columns: numerical_features.append(f'{player}_active_hp_perc')
            if f'{player}_active_boost_atk' in X.columns: numerical_features.append(f'{player}_active_boost_atk')
            if f'{player}_active_boost_def' in X.columns: numerical_features.append(f'{player}_active_boost_def')
            if f'{player}_active_boost_spa' in X.columns: numerical_features.append(f'{player}_active_boost_spa')
            if f'{player}_active_boost_spd' in X.columns: numerical_features.append(f'{player}_active_boost_spd')
            if f'{player}_active_boost_spe' in X.columns: numerical_features.append(f'{player}_active_boost_spe')
            if f'{player}_active_terastallized' in X.columns: numerical_features.append(f'{player}_active_terastallized')
        numerical_features.extend(new_binary_move_cols)
        for i in range(1, 7):
            for player in ['p1', 'p2']:
                 hp_col, status_col, species_col = f'{player}_slot{i}_hp_perc', f'{player}_slot{i}_status', f'{player}_slot{i}_species'
                 if hp_col in X.columns: numerical_features.append(hp_col)
                 if status_col in X.columns: categorical_features.append(status_col)
                 if species_col in X.columns: categorical_features.append(species_col)
        categorical_features.extend([f for f in field_cols if f in X.columns])
        numerical_features.extend([f for f in hazard_cols if f in X.columns])
        numerical_features.extend([f for f in side_cond_cols if f in X.columns])
        if 'last_move_p1' in X.columns: categorical_features.append('last_move_p1')
        if 'last_move_p2' in X.columns: categorical_features.append('last_move_p2')
        if 'turn_number' in X.columns: numerical_features.append('turn_number')
        if 'predicted_winner' in X.columns: categorical_features.append('predicted_winner')
        all_medium_cols = list(X.columns)
        numerical_features = sorted(list(set([f for f in numerical_features if f in all_medium_cols])))
        categorical_features = sorted(list(set([f for f in categorical_features if f in all_medium_cols])))
        overlap = set(numerical_features) & set(categorical_features)
        if overlap:
            print(f"Warning: Overlap detected: {overlap}. Removing from numerical.")
            numerical_features = [f for f in numerical_features if f not in overlap]
        # --- End feature types ---


    elif feature_set == 'full':
        # (Keep V4 'full' logic - multi-hot encodes all revealed_moves columns)
        print("\n--- Using FULL feature set ---")
        print("\nPreparing features and target...")
        base_exclude = ['replay_id', 'action_taken', 'battle_winner']
        if 'player_to_move' in df.columns: base_exclude.append('player_to_move') # Exclude if present
        cols_to_exclude = base_exclude
        feature_columns = [col for col in df.columns if col not in cols_to_exclude]
        X = df[feature_columns].copy()
        print(f"Initial feature count: {len(feature_columns)}")

        print("\nProcessing 'revealed_moves' features (Multi-Hot Encoding - ALL SLOTS)...")
        revealed_move_cols = sorted([col for col in X.columns if col.endswith('_revealed_moves')])
        all_revealed_moves = set()
        new_binary_move_cols = []

        if not revealed_move_cols:
            print("Warning: No '*_revealed_moves' columns found.")
        else:
            print("  Finding unique revealed moves...")
            for col in revealed_move_cols:
                unique_in_col = X[col].fillna('').astype(str).str.split(',').explode().unique()
                all_revealed_moves.update(m for m in unique_in_col if m and m != 'none' and m != 'error_state')
            unique_moves_list = sorted(list(all_revealed_moves))
            print(f"  Found {len(unique_moves_list)} unique revealed moves across all slots.")
            print("  Creating and populating binary revealed move columns...")
            for base_col in revealed_move_cols:
                X[base_col] = X[base_col].fillna('none')
                try:
                    revealed_sets = X[base_col].str.split(',').apply(set)
                    for move in unique_moves_list:
                        # Sanitize move name
                        sanitized_move_name = move.replace(' ', '_').replace('-', '_').replace(':', '').replace('%', 'perc')
                        new_col_name = f"{base_col}_{sanitized_move_name}"
                        X[new_col_name] = revealed_sets.apply(lambda move_set: 1 if move in move_set else 0).astype(np.int8)
                        new_binary_move_cols.append(new_col_name)
                    del revealed_sets
                except Exception as e:
                    print(f"  Error processing revealed moves in column {base_col}: {e}")
                gc.collect()
            print(f"  Created {len(new_binary_move_cols)} new binary move features.")
            print("  Dropping original revealed_moves string columns...")
            X = X.drop(columns=revealed_move_cols)
            gc.collect()

        print("\nIdentifying final feature types and handling remaining NaNs for 'full' set...")
        if 'last_move_p1' in X.columns: X['last_move_p1'] = X['last_move_p1'].fillna('none').astype('category')
        if 'last_move_p2' in X.columns: X['last_move_p2'] = X['last_move_p2'].fillna('none').astype('category')
        obj_cols = X.select_dtypes(include=['object']).columns.tolist()
        for col in obj_cols: X[col] = X[col].fillna('Unknown').astype('category') # Use Unknown consistently

        # Add new binary move cols to numerical
        numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        numerical_features.extend(new_binary_move_cols)
        numerical_features = sorted(list(set(numerical_features))) # Ensure unique and sorted

        categorical_features = sorted(list(set(X.select_dtypes(include=['category']).columns.tolist())))

        overlap = set(numerical_features) & set(categorical_features)
        if overlap: numerical_features = [f for f in numerical_features if f not in overlap]

        if numerical_features:
             nan_counts = X[numerical_features].isnull().sum()
             cols_with_nan = nan_counts[nan_counts > 0].index.tolist()
             if cols_with_nan:
                 print(f"  Numerical columns have NaNs: {cols_with_nan}. Filling with median.")
                 for col in cols_with_nan:
                      median_val = X[col].median()
                      fill_value = median_val if pd.notna(median_val) else 0
                      # Use direct assignment which is less prone to SettingWithCopyWarning
                      X[col] = X[col].fillna(fill_value)

        if X.isnull().sum().sum() > 0:
            print("Warning: NaNs still present after handling. Forcing fill.")
            for col in X.columns:
                 if X[col].isnull().any():
                      if pd.api.types.is_numeric_dtype(X[col]):
                          X[col] = X[col].fillna(0) # Use assignment
                      else:
                          # Ensure categorical NaNs are filled correctly
                          if pd.api.types.is_categorical_dtype(X[col]):
                              if 'Unknown' not in X[col].cat.categories:
                                   X[col] = X[col].cat.add_categories(['Unknown'])
                              X[col] = X[col].fillna('Unknown') # Use assignment
                          else: # If somehow still object/other
                              X[col] = X[col].fillna('Unknown') # Use assignment


    else:
        print(f"Error: Invalid feature_set '{feature_set}'. Choose 'full', 'medium', or 'simplified'.")
        return

    # --- Fill NaNs (General Check - applied AFTER feature set specific handling) ---
    # Note: This section might be slightly redundant now with the improved NaN handling
    # within the feature set blocks, but can serve as a final check.
    print(f"\nFinal NaN Check for '{feature_set}' set...")
    nan_report_before = X.isnull().sum()
    cols_with_nan_before = nan_report_before[nan_report_before > 0]
    if not cols_with_nan_before.empty:
        print(f"  NaNs found BEFORE final handling in columns: {cols_with_nan_before.index.tolist()}")
        # Re-identify types just in case columns were added/changed
        final_numerical = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
        final_categorical = [col for col in X.columns if pd.api.types.is_categorical_dtype(X[col]) or X[col].dtype == 'object']

        for col in final_numerical:
            if X[col].isnull().any():
                 median_val = X[col].median()
                 fill_value = median_val if pd.notna(median_val) else 0
                 X[col] = X[col].fillna(fill_value) # Use assignment

        default_cat_fill = 'Unknown'
        for col in final_categorical:
             if X[col].isnull().any():
                  # Ensure 'Unknown' category exists if categorical
                  if pd.api.types.is_categorical_dtype(X[col]):
                      if default_cat_fill not in X[col].cat.categories:
                           X[col] = X[col].cat.add_categories([default_cat_fill])
                      X[col] = X[col].fillna(default_cat_fill) # Use assignment
                  else: # Object type
                       X[col] = X[col].fillna(default_cat_fill).astype('category') # Fill and convert
        print("  Final NaN handling complete.")
    else:
        print("  No NaNs found before final encoding/scaling.")

    # Final Check
    nan_report_after = X.isnull().sum()
    if nan_report_after.sum() > 0:
        print("Error: NaNs still present AFTER final handling. Columns:")
        print(nan_report_after[nan_report_after > 0])
        return


    # --- Common Steps from here ---

    print(f"\nFinal feature counts for '{feature_set}' set:")
    # Use the feature lists determined by the specific feature_set logic
    print(f"  Numerical: {len(numerical_features)}")
    print(f"  Categorical: {len(categorical_features)}")
    print(f"  Total Features in X: {X.shape[1]}")


    # --- Encode Target Variable (y) ---
    print(f"\nEncoding target variable ({predict_mode})...") # <--- MODIFIED print
    label_encoder = LabelEncoder()
    try:
        # Ensure y_raw (which depends on predict_mode) is string
        y_encoded = label_encoder.fit_transform(y_raw.astype(str))
    except Exception as e:
        print(f"Error encoding target: {e}. Check target content.")
        return
    num_classes = len(label_encoder.classes_)
    print(f"Found {num_classes} unique actions/moves in the target set.")
    if num_classes < 2: print("Error: Need at least 2 unique actions/moves."); return

    # <--- MODIFIED: Use label_encoder_suffix in filename --->
    label_encoder_path = f'action_label_encoder_v4_{label_encoder_suffix}.joblib'
    joblib.dump(label_encoder, label_encoder_path)
    print(f"Label encoder saved to {label_encoder_path}")
    del y_raw; gc.collect()


    # --- Split Data ---
    print("\nSplitting data into Train, Validation, Test sets...")
    try:
        # Stratify only if feasible
        stratify_option = y_encoded if num_classes < 1000 and len(np.unique(y_encoded)) > 1 else None
        min_samples_per_class = np.min(np.bincount(y_encoded))
        can_stratify = stratify_option is not None and min_samples_per_class >= 2

        if can_stratify:
             X_train_full, X_test, y_train_full_encoded, y_test_encoded = train_test_split(
                 X, y_encoded, test_size=test_split_size, random_state=42, stratify=y_encoded
             )
             train_full_size = 1.0 - test_split_size
             val_size_relative = val_split_size / train_full_size if train_full_size > 0 else 0.1
             if not (0 < val_size_relative < 1): val_size_relative = 0.15 # Adjust relative size calculation

             # Check stratification for second split
             min_samples_train_full = np.min(np.bincount(y_train_full_encoded)) if len(y_train_full_encoded) > 0 else 0
             can_stratify_2 = min_samples_train_full >= 2

             if can_stratify_2:
                 X_train, X_val, y_train_encoded, y_val_encoded = train_test_split(
                     X_train_full, y_train_full_encoded, test_size=val_size_relative, random_state=42, stratify=y_train_full_encoded
                 )
                 print("Successfully split data WITH stratification.")
             else:
                  print("Warning: Cannot stratify train/val split due to rare classes in train_full set. Splitting randomly.")
                  X_train, X_val, y_train_encoded, y_val_encoded = train_test_split(
                      X_train_full, y_train_full_encoded, test_size=val_size_relative, random_state=42
                  )
        else:
             print(f"Warning: Cannot stratify train/test split (min samples/class: {min_samples_per_class}, num_classes: {num_classes}). Splitting randomly.")
             X_train_full, X_test, y_train_full_encoded, y_test_encoded = train_test_split(
                 X, y_encoded, test_size=test_split_size, random_state=42
             )
             train_full_size = 1.0 - test_split_size
             val_size_relative = val_split_size / train_full_size if train_full_size > 0 else 0.1
             if not (0 < val_size_relative < 1): val_size_relative = 0.15
             X_train, X_val, y_train_encoded, y_val_encoded = train_test_split(
                 X_train_full, y_train_full_encoded, test_size=val_size_relative, random_state=42
             )

    except ValueError as e: # Catch other potential split errors
        print(f"Error during data splitting: {e}. Splitting randomly as fallback.")
        # Fallback to random split
        X_train_full, X_test, y_train_full_encoded, y_test_encoded = train_test_split(
            X, y_encoded, test_size=test_split_size, random_state=42
        )
        train_full_size = 1.0 - test_split_size
        val_size_relative = val_split_size / train_full_size if train_full_size > 0 else 0.1
        if not (0 < val_size_relative < 1): val_size_relative = 0.15
        X_train, X_val, y_train_encoded, y_val_encoded = train_test_split(
            X_train_full, y_train_full_encoded, test_size=val_size_relative, random_state=42
        )
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")
    del X, df, X_train_full, y_train_full_encoded; gc.collect()


    # --- Calculate Class Weights ---
    print("\nCalculating class weights for handling imbalance...")
    if len(y_train_encoded) == 0: print("Error: y_train_encoded empty."); return
    unique_train_classes, class_counts = np.unique(y_train_encoded, return_counts=True)
    if len(unique_train_classes) < 2:
         print("Warning: Fewer than 2 classes in training data. Using uniform weights.")
         class_weight_dict = {cls_idx: 1.0 for cls_idx in range(num_classes)}
    else:
        try:
             class_weights_values = compute_class_weight('balanced', classes=unique_train_classes, y=y_train_encoded)
             class_weight_dict = dict(zip(unique_train_classes, class_weights_values))
             # Add weight 1.0 for classes not seen in training (essential for TF)
             all_possible_classes = np.arange(num_classes)
             for cls_idx in all_possible_classes:
                  if cls_idx not in class_weight_dict:
                      # print(f"Debug: Assigning weight 1.0 to class {cls_idx} (not in train set)") # Optional Debug
                      class_weight_dict[cls_idx] = 1.0 # Assign default weight
             print(f"Class weights calculated (Example: {list(class_weight_dict.items())[:5]}...)")
        except Exception as e:
             print(f"Error calculating class weights: {e}. Using uniform weights.")
             class_weight_dict = {cls_idx: 1.0 for cls_idx in range(num_classes)}


    # --- Preprocess X data based on model type ---
    X_train_processed, X_val_processed, X_test_processed = None, None, None
    preprocessor = None
    # <--- MODIFIED: Use label_encoder_suffix in filenames --->
    feature_lists_path = f'action_feature_lists_v4_{label_encoder_suffix}.joblib'
    preprocessor_path = f'action_tf_preprocessor_v4_{label_encoder_suffix}.joblib'

    # Use the actual columns present in X_train after feature engineering
    final_train_cols = X_train.columns.tolist()
    # Ensure numerical/categorical feature lists only contain columns present in X_train
    numerical_features = [f for f in numerical_features if f in final_train_cols]
    categorical_features = [f for f in categorical_features if f in final_train_cols]
    try:
         joblib.dump({
             'feature_columns_final': final_train_cols,
             'numerical_features': numerical_features, # Use filtered lists
             'categorical_features': categorical_features # Use filtered lists
             }, feature_lists_path)
         print(f"Final feature lists saved to {feature_lists_path}")
    except Exception as e: print(f"Error saving feature lists: {e}")


    if model_type == 'tensorflow':
        print("\nSetting up TF preprocessing pipeline (OneHotEncoder + Scaler)...")
        transformers = []
        # Use the potentially filtered numerical_features list
        if numerical_features: # Check if list is not empty
            numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
            transformers.append(('num', numerical_transformer, numerical_features))
        else: print("Info: No numerical features to scale for TF.")
        # Use the potentially filtered categorical_features list
        if categorical_features: # Check if list is not empty
            categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))])
            transformers.append(('cat', categorical_transformer, categorical_features))
        else: print("Info: No categorical features to OneHotEncode for TF.")

        if not transformers: print("Error: No transformers created for TF!"); return

        preprocessor = ColumnTransformer(transformers=transformers, remainder='drop', sparse_threshold=0.3)

        print("Applying TF preprocessing (fit on train, transform all)...")
        try:
            X_train_processed = preprocessor.fit_transform(X_train)
            print(f"  Fit TF preprocessor on training data (Output shape: {X_train_processed.shape})")
            X_val_processed = preprocessor.transform(X_val)
            X_test_processed = preprocessor.transform(X_test)
            print(f"TF Processed shapes - Train: {X_train_processed.shape}, Val: {X_val_processed.shape}, Test: {X_test_processed.shape}")
            joblib.dump(preprocessor, preprocessor_path)
            print(f"TF preprocessor saved to {preprocessor_path}")
        except ValueError as e:
            print(f"ValueError during TF preprocessing: {e}")
            # Debug mismatch
            if categorical_features:
                for col in categorical_features:
                    if col in X_train.columns and col in X_val.columns and col in X_test.columns:
                         train_cats = set(X_train[col].unique())
                         val_cats = set(X_val[col].unique())
                         test_cats = set(X_test[col].unique())
                         if not val_cats.issubset(train_cats) or not test_cats.issubset(train_cats):
                              print(f"  Mismatch detected in column '{col}':")
                              print(f"    Val not in Train: {val_cats - train_cats}")
                              print(f"    Test not in Train: {test_cats - train_cats}")
                    else: print(f"  Column '{col}' not present in all splits.")
            return
        except MemoryError: print("MemoryError during TF preprocessing."); return
        except Exception as e: print(f"Error during TF preprocessing: {e}"); import traceback; traceback.print_exc(); return

        del X_train, X_val, X_test; gc.collect()

    elif model_type == 'lightgbm':
        print("\nPreprocessing for LGBM (dtype conversion, scaling) will occur inside its training function.")
        X_train_processed, X_val_processed, X_test_processed = X_train, X_val, X_test # Pass original DFs

    else:
        print(f"Error: Unknown model_type '{model_type}'"); return

    # --- Train Selected Model ---
    print(f"\n--- Initiating {model_type.upper()} Model Training ({feature_set.upper()} features, {predict_mode.upper()}) ---") # <--- MODIFIED print
    if model_type == 'tensorflow':
         if X_train_processed is not None:
             train_tensorflow_action_predictor(X_train_processed, X_val_processed, X_test_processed,
                                               y_train_encoded, y_val_encoded, y_test_encoded,
                                               num_classes, class_weight_dict, label_encoder,
                                               epochs, batch_size, learning_rate,
                                               label_suffix=label_encoder_suffix) # <--- MODIFIED pass suffix
         else: print("Skipping TF training due to preprocessing errors.")
    elif model_type == 'lightgbm':
         # Pass potentially filtered feature lists
         train_lgbm_action_predictor(X_train_processed, X_val_processed, X_test_processed,
                                     y_train_encoded, y_val_encoded, y_test_encoded,
                                     numerical_features, categorical_features,
                                     num_classes, class_weight_dict, label_encoder,
                                     label_suffix=label_encoder_suffix) # <--- MODIFIED pass suffix


# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train action predictor (V4 with predict_mode).") # <--- MODIFIED description
    parser.add_argument("parquet_file", type=str, help="Path to the input Parquet file.")
    parser.add_argument("--model_type", choices=['tensorflow', 'lightgbm'], default='lightgbm', help="Type of model to train.")
    parser.add_argument("--feature_set", choices=['full', 'medium', 'simplified'], default='full', help="Feature set to use.")
    # <--- NEW Argument --->
    parser.add_argument("--predict_mode", choices=['move_only', 'all_actions'], default='move_only',
                        help="Predict only moves or all actions (default: move_only).")
    # --------------------
    parser.add_argument("--min_turn", type=int, default=1, help="Minimum turn number to include.")
    parser.add_argument("--test_split", type=float, default=0.2, help="Fraction for test set.")
    parser.add_argument("--val_split", type=float, default=0.15, help="Fraction for validation set.")
    # TF specific args
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs (TF only).")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size (TF only).")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (TF only).")

    args = parser.parse_args()

    # Validate split sizes
    if args.test_split + args.val_split >= 1.0: print("Error: test_split + val_split must be < 1.0"); exit(1)
    if args.test_split <= 0 or args.val_split <= 0: print("Error: test_split and val_split must be > 0."); exit(1)

    # <--- MODIFIED call to pass new argument --->
    run_action_training(
        parquet_path=args.parquet_file,
        model_type=args.model_type,
        feature_set=args.feature_set,
        predict_mode=args.predict_mode, # Pass the new mode
        min_turn=args.min_turn,
        test_split_size=args.test_split,
        val_split_size=args.val_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
