import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import argparse
import os
import gc
from collections import Counter
import warnings

# Suppress PerformanceWarnings temporarily if desired, though fixing the cause is better
# warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# --- Helper Function (find_active_species - unchanged) ---
def find_active_species(row, player_prefix):
    for i in range(1, 7):
        active_col = f"{player_prefix}_slot{i}_is_active"
        species_col = f"{player_prefix}_slot{i}_species"
        if active_col in row.index and species_col in row.index:
            if row[active_col] == 1:
                return row[species_col] if pd.notna(row[species_col]) else 'Unknown'
    return 'Unknown'

# --- Function to Load Artifacts (unchanged) ---
def load_artifacts(model_path, info_path, scaler_path, encoder_path):
    try:
        print(f"Loading LGBM model from: {model_path}")
        lgbm_model = lgb.Booster(model_file=model_path)
        print(f"Loading feature info from: {info_path}")
        feature_info = joblib.load(info_path)
        print(f"Loading scaler from: {scaler_path}")
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        if scaler: print("Scaler loaded.")
        else: print("Scaler not found or not applicable.")
        print(f"Loading label encoder from: {encoder_path}")
        label_encoder = joblib.load(encoder_path)
        print("Artifacts loaded successfully.")
        return lgbm_model, feature_info, scaler, label_encoder
    except FileNotFoundError as e: print(f"Error loading artifacts: {e}. Check paths."); exit(1)
    except Exception as e: print(f"Unexpected error loading artifacts: {e}"); exit(1)

# --- Function to Prepare Input Data (Refined) ---
def prepare_input_data_medium(df_input_raw, feature_info, scaler):
    print("Preparing input data using 'medium' feature set logic...")
    X = df_input_raw.copy()

    # --- Feature Engineering ---
    base_active_features = ['species', 'hp_perc', 'status', 'boost_atk', 'boost_def', 'boost_spa', 'boost_spd', 'boost_spe', 'terastallized', 'tera_type']
    active_data_list = []
    print("  Extracting active Pokemon details...")
    # (Active data extraction loop - unchanged from previous version)
    for idx, row in X.iterrows():
        active_p1_slot, active_p2_slot = -1, -1
        for i_slot in range(1, 7):
            if row.get(f'p1_slot{i_slot}_is_active', 0) == 1: active_p1_slot = i_slot
            if row.get(f'p2_slot{i_slot}_is_active', 0) == 1: active_p2_slot = i_slot
        row_active_data = {'original_index': idx}
        # P1
        p1_active_moves_str = 'none'
        if active_p1_slot != -1:
            for feat in base_active_features: row_active_data[f'p1_active_{feat}'] = row.get(f'p1_slot{active_p1_slot}_{feat}', None)
            p1_moves_col_name = f'p1_slot{active_p1_slot}_revealed_moves'; p1_active_moves_str = row.get(p1_moves_col_name, 'none') if p1_moves_col_name in row.index else 'none'
        else:
            for feat in base_active_features: row_active_data[f'p1_active_{feat}'] = None
        row_active_data['p1_active_revealed_moves_str'] = p1_active_moves_str
        # P2
        p2_active_moves_str = 'none'
        if active_p2_slot != -1:
            for feat in base_active_features: row_active_data[f'p2_active_{feat}'] = row.get(f'p2_slot{active_p2_slot}_{feat}', None)
            p2_moves_col_name = f'p2_slot{active_p2_slot}_revealed_moves'; p2_active_moves_str = row.get(p2_moves_col_name, 'none') if p2_moves_col_name in row.index else 'none'
        else:
            for feat in base_active_features: row_active_data[f'p2_active_{feat}'] = None
        row_active_data['p2_active_revealed_moves_str'] = p2_active_moves_str
        active_data_list.append(row_active_data)
    # (End of active data extraction loop)

    active_df = pd.DataFrame(active_data_list).set_index('original_index')
    active_df['p1_active_revealed_moves_str'] = active_df['p1_active_revealed_moves_str'].fillna('none').astype(str)
    active_df['p2_active_revealed_moves_str'] = active_df['p2_active_revealed_moves_str'].fillna('none').astype(str)

    all_model_features = feature_info.get('feature_names_in_order', [])
    if not all_model_features: print("Error: 'feature_names_in_order' missing."); exit(1)
    base_cols_to_keep = [
        col for col in df_input_raw.columns if col in all_model_features and not col.startswith('p1_active_')
        and not col.startswith('p2_active_') and not col.endswith('_revealed_moves')
        and col not in ['p1_active_revealed_moves_str', 'p2_active_revealed_moves_str'] # Explicitly exclude temp cols
    ]
    # Ensure no overlap between base cols and active_df columns before concat
    active_df_cols = active_df.columns.tolist()
    base_cols_to_keep = [col for col in base_cols_to_keep if col not in active_df_cols]
    X_processed = pd.concat([X[base_cols_to_keep], active_df], axis=1)
    print(f"  Shape after adding active details: {X_processed.shape}")

    # Multi-Hot Encode Active Revealed Moves (Refined for Performance)
    print("  Processing active revealed moves...")
    active_revealed_move_cols = ['p1_active_revealed_moves_str', 'p2_active_revealed_moves_str']
    known_moves = set()
    prefix1, prefix2 = "p1_active_revealed_move_", "p2_active_revealed_move_"
    numerical_trained = feature_info.get('numerical_features', [])
    for col_name in numerical_trained: # Infer known moves from trained feature names
        move_name = None
        if col_name.startswith(prefix1): move_name = col_name[len(prefix1):]
        elif col_name.startswith(prefix2): move_name = col_name[len(prefix2):]
        if move_name: known_moves.add(move_name.replace('_', ' ')) # Use original-like name

    if not known_moves:
        print("  Warning: Could not determine known moves.")
        new_move_cols_df = pd.DataFrame(index=X_processed.index) # Empty df if no moves
    else:
        print(f"  Found {len(known_moves)} known moves from training features.")
        known_moves_list = sorted(list(known_moves))
        new_move_cols_dict = {} # Collect new columns here

        for base_col in active_revealed_move_cols:
            if base_col not in X_processed.columns: continue
            player_prefix = base_col.split('_')[0]
            new_col_prefix = f"{player_prefix}_active_revealed_move"
            revealed_sets = X_processed[base_col].fillna('none').astype(str).str.split(',').apply(set)

            for move in known_moves_list:
                sanitized_move_name = move.replace(' ', '_').replace('-', '_').replace(':', '').replace('%', 'perc')
                expected_col_name = f"{new_col_prefix}_{sanitized_move_name}"
                if expected_col_name in all_model_features:
                    # Store Series in dict instead of assigning directly to X_processed
                    new_move_cols_dict[expected_col_name] = revealed_sets.apply(lambda move_set: 1 if move in move_set else 0).astype(np.int8)
            del revealed_sets
        gc.collect()
        # Create DataFrame from collected columns
        new_move_cols_df = pd.DataFrame(new_move_cols_dict, index=X_processed.index)
        print(f"  Created {len(new_move_cols_df.columns)} new binary active move features.")

    # Concat new move columns at once (fixes fragmentation warning)
    X_processed = pd.concat([X_processed, new_move_cols_df], axis=1)

    # Drop the temporary string columns
    cols_to_drop = [col for col in active_revealed_move_cols if col in X_processed.columns]
    if cols_to_drop:
        X_processed = X_processed.drop(columns=cols_to_drop)
    print(f"  Shape after active move encoding: {X_processed.shape}")
    # Defragment after major additions (optional but can help)
    X_processed = X_processed.copy()
    gc.collect()

    # --- Preprocessing ---

    # Handle NaNs
    print("  Handling potential NaNs...")
    numerical_trained = feature_info.get('numerical_features', [])
    categorical_trained = feature_info.get('categorical_features', [])
    category_map = feature_info.get('category_map', {})
    default_cat_fill = 'Unknown'

    for col in numerical_trained:
        if col in X_processed.columns and X_processed[col].isnull().any():
            fill_value = 0
            if col.endswith('_hp_perc'): fill_value = 100
            X_processed[col] = X_processed[col].fillna(fill_value) # Use direct assignment

    for col in categorical_trained:
         if col in X_processed.columns:
              # Fill NaNs first
              if X_processed[col].isnull().any():
                   X_processed[col] = X_processed[col].fillna(default_cat_fill) # Assignment
              # Ensure string type before applying category map
              X_processed[col] = X_processed[col].astype(str)
              # Apply known categories
              if col in category_map:
                  known_categories = category_map[col].categories.tolist()
                  # Add 'Unknown' if it's not already known for this specific column
                  if default_cat_fill not in known_categories:
                       known_categories.append(default_cat_fill)
                  # Create the dtype *with* Unknown included
                  cat_type = pd.CategoricalDtype(categories=known_categories, ordered=False)
                  try:
                       X_processed[col] = X_processed[col].astype(cat_type)
                  except ValueError as e:
                       print(f"    Warning: Error setting dtype for {col}: {e}. Might contain unseen values not handled.")
                       # Fallback: just ensure it's category, might miss some known cats
                       X_processed[col] = X_processed[col].astype('category')

                  # Check for NaNs *after* applying dtype (unseen values become NaN)
                  if X_processed[col].isnull().any():
                       print(f"    Filling NaNs from unseen categories in '{col}'")
                       # Re-apply category dtype after fillna if needed, ensures 'Unknown' is valid
                       X_processed[col] = X_processed[col].fillna(default_cat_fill).astype(cat_type)
              else:
                   print(f"    Warning: No category map found for '{col}'. Basic category conversion.")
                   X_processed[col] = X_processed[col].astype('category')
         # else: print(f"Debug: Categorical column '{col}' not found in input.") # Optional

    # Scale Numerical Features
    features_to_scale = feature_info.get('features_scaled')
    if features_to_scale is None: features_to_scale = [f for f in numerical_trained if f in X_processed.columns]
    if scaler and features_to_scale:
        print(f"  Scaling {len(features_to_scale)} numerical features...")
        cols_to_scale_present = [col for col in features_to_scale if col in X_processed.columns]
        if cols_to_scale_present:
             X_processed[cols_to_scale_present] = scaler.transform(X_processed[cols_to_scale_present])
        else: print("    Warning: None of the features to scale were found.")
    elif scaler: print("  Scaler loaded, but no features identified for scaling.")
    else: print("  No scaler loaded or needed.")

    # Ensure Column Order and Subset + Add Missing
    print("  Ensuring final column order and subset...")
    final_feature_names = feature_info.get('feature_names_in_order', [])
    if not final_feature_names: print("Error: Missing 'feature_names_in_order'."); return None

    missing_cols = set(final_feature_names) - set(X_processed.columns)
    if missing_cols:
        print(f"  Warning: Input data missing {len(missing_cols)} columns expected. Adding with default 0.")
        missing_cols_df = pd.DataFrame(0, index=X_processed.index, columns=list(missing_cols))
        # Concat missing cols at once
        X_processed = pd.concat([X_processed, missing_cols_df], axis=1)

    # *** CRITICAL FIX for Categorical Mismatch ***
    # Ensure ALL columns designated as categorical during training ARE category dtype NOW
    print("  Verifying categorical feature dtypes...")
    final_categorical_trained = [f for f in categorical_trained if f in final_feature_names] # Only care about final cols
    error_found = False
    for col in final_categorical_trained:
        if col not in X_processed.columns:
             print(f"    ERROR: Categorical column '{col}' expected but missing even after adding defaults!")
             error_found = True
             continue
        if not pd.api.types.is_categorical_dtype(X_processed[col]):
             print(f"    WARNING: Column '{col}' was categorical in training but is now {X_processed[col].dtype}. Attempting conversion.")
             try:
                 # Attempt conversion back to category; might fail if map was missing etc.
                 X_processed[col] = X_processed[col].astype('category')
             except Exception as e:
                 print(f"      ERROR: Failed to convert '{col}' back to category: {e}")
                 error_found = True
    if error_found:
        print("  ERROR: Cannot proceed due to missing/unconvertible categorical columns.")
        return None
    # --- End Critical Fix ---


    # Select and reorder columns
    try:
        # Use .copy() to help avoid fragmentation and ensure clean final DataFrame
        X_final = X_processed[final_feature_names].copy()
    except KeyError as e:
        print(f"Error: Could not select all expected columns. Missing: {e}")
        still_missing = set(final_feature_names) - set(X_processed.columns)
        print(f"Columns still missing: {still_missing}")
        return None

    print(f"Final processed shape for prediction: {X_final.shape}")
    return X_final

# --- Main Prediction Function (Unchanged) ---
def predict_moves(df_input, model, feature_info, scaler, label_encoder, top_k=5):
    X_processed = prepare_input_data_medium(df_input, feature_info, scaler)
    if X_processed is None: print("Error during data preparation."); return None

    print("\nMaking predictions...")
    try:
        # --- Add check for categorical features BEFORE prediction ---
        categorical_features_for_lgbm = [f for f in feature_info.get('categorical_features', []) if f in X_processed.columns]
        print(f"  Passing {len(categorical_features_for_lgbm)} features as categorical to LGBM.")
        # print(f"  Categorical feature sample: {categorical_features_for_lgbm[:10]}") # Debug
        # Ensure X_processed dtypes are correct here
        # print(f"  Data types before predict:\n{X_processed.dtypes.value_counts()}") # Debug

        pred_probabilities = model.predict(X_processed, categorical_feature=categorical_features_for_lgbm) # Specify categoricals
    except Exception as e:
         print(f"Error during model prediction: {e}")
         print(f"Model expected features: {feature_info.get('feature_names_in_order')[:10]}...")
         print(f"Actual columns passed: {X_processed.columns.tolist()[:10]}...")
         print(f"Data shape passed: {X_processed.shape}")
         print(f"Data types:\n{X_processed.dtypes.value_counts()}")
         # Check for mismatch explicitly
         lgbm_cats = set(model.feature_name()) & set(categorical_features_for_lgbm)
         passed_cats = set(X_processed.select_dtypes(include='category').columns)
         if lgbm_cats != passed_cats:
              print(f"Mismatch Detail: Expected Cats {len(lgbm_cats)}, Passed Cats {len(passed_cats)}")
              # print(f"Missing from passed: {lgbm_cats - passed_cats}")
              # print(f"Extra in passed: {passed_cats - lgbm_cats}")

         return None

    print("Decoding predictions...")
    predictions = []
    for i in range(len(pred_probabilities)):
        row_probs = pred_probabilities[i]
        top_k_indices = np.argsort(row_probs)[-top_k:][::-1]
        top_k_moves = label_encoder.inverse_transform(top_k_indices)
        top_k_probs = row_probs[top_k_indices]
        result = {'rank': list(range(1, top_k + 1)), 'move': top_k_moves, 'probability': top_k_probs}
        if df_input.index is not None and i < len(df_input.index): result['original_index'] = df_input.index[i]
        predictions.append(pd.DataFrame(result))
    return predictions


# --- Main Execution Block (Unchanged) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Pokemon move using LightGBM.")
    parser.add_argument("input_parquet", type=str, help="Path to input Parquet file.")
    parser.add_argument("--model_path", type=str, default="action_lgbm_model_v4_medium_move_only.txt", help="Path to LGBM model.")
    parser.add_argument("--info_path", type=str, default="action_lgbm_feature_info_v4_medium_move_only.joblib", help="Path to feature info.")
    parser.add_argument("--scaler_path", type=str, default="action_lgbm_scaler_v4_medium_move_only.joblib", help="Path to scaler.")
    parser.add_argument("--encoder_path", type=str, default="action_label_encoder_v4_medium_move_only.joblib", help="Path to label encoder.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of predictions.")
    args = parser.parse_args()

    print(f"Loading input data from: {args.input_parquet}")
    try:
        input_df = pd.read_parquet(args.input_parquet)
        print(f"Loaded {len(input_df)} game state(s).")
        if 'turn_number' not in input_df.columns: print("Warning: Missing 'turn_number'.")
    except FileNotFoundError: print(f"Error: Input file not found: {args.input_parquet}"); exit(1)
    except Exception as e: print(f"Error loading input: {e}"); exit(1)

    model, feature_info, scaler, label_encoder = load_artifacts(
        args.model_path, args.info_path, args.scaler_path, args.encoder_path
    )

    predictions_list = predict_moves(input_df, model, feature_info, scaler, label_encoder, args.top_k)

    if predictions_list:
        print(f"\n--- Top {args.top_k} Predicted Moves ---")
        for i, pred_df in enumerate(predictions_list):
            idx_info = f"Input Row Index: {pred_df['original_index'].iloc[0]}" if 'original_index' in pred_df else f"Input Row {i}"
            print(f"\nPrediction for {idx_info}:")
            pred_df['probability'] = pred_df['probability'].map('{:.2%}'.format)
            print(pred_df[['rank', 'move', 'probability']].to_string(index=False))
    else:
        print("\nNo predictions generated due to errors.")

    print("\nPrediction script finished.")