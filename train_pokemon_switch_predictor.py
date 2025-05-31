import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, top_k_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
# from sklearn.base import BaseEstimator, TransformerMixin # Not used in this version, can be removed
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

# --- Helper Function (find_active_species - unchanged from train_action_predictor.py) ---
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

# --- TF Training Function (MODIFIED for switch prediction) ---
def train_tensorflow_switch_predictor(X_train_processed, X_val_processed, X_test_processed,
                                      y_train_encoded, y_val_encoded, y_test_encoded,
                                      num_classes, class_weight_dict, label_encoder,
                                      epochs=20, batch_size=128, learning_rate=0.001,
                                      label_suffix=""): # Suffix for unique filenames
    print(f"\n--- Training TensorFlow Switch Model ---")
    print(f"Input shape: {X_train_processed.shape[1]}")
    print(f"Num classes (target species): {num_classes}")

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

    input_dim = X_train_processed.shape[1]
    print(f"Building TF model with input dimension: {input_dim}")
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, use_bias=False), BatchNormalization(), Activation('relu'), Dropout(0.3),
        Dense(128, use_bias=False), BatchNormalization(), Activation('relu'), Dropout(0.3),
        Dense(64, use_bias=False), BatchNormalization(), Activation('relu'), Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_10_accuracy')]) # k=3 as there are ~5 options
    model.summary()

    print("\nStarting TF model training...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    history = model.fit(
        X_train_processed, y_train_one_hot,
        validation_data=(X_val_processed, y_val_one_hot),
        epochs=epochs, batch_size=batch_size, class_weight=class_weight_dict,
        callbacks=[early_stopping], verbose=2
    )
    print("TF Training finished.")

    print("\nEvaluating TF model on the test set...")
    results = model.evaluate(X_test_processed, y_test_one_hot, verbose=0)
    loss, accuracy = results[0], results[1]
    top_k_acc = results[2] if len(results) > 2 else np.nan
    print(f"TF Test Loss: {loss:.4f}")
    print(f"TF Test Accuracy: {accuracy:.4f}")
    print(f"TF Test Top-3 Accuracy: {top_k_acc:.4f}")

    model_save_path = f'switch_tf_model_v1_{label_suffix}.keras'
    print(f"Saving TF model to {model_save_path}")
    try:
        model.save(model_save_path)
        print("TF Model saved.")
    except Exception as e:
        print(f"Error saving TF model: {e}")
    return history, model

# --- LGBM Training Function (MODIFIED for switch prediction) ---
def train_lgbm_switch_predictor(X_train, X_val, X_test,
                                y_train_encoded, y_val_encoded, y_test_encoded,
                                numerical_features, categorical_features,
                                num_classes, class_weight_dict, label_encoder,
                                label_suffix=""): # Suffix for unique filenames
    print(f"\n--- Training LightGBM Switch Model ---")
    print(f"Using {len(numerical_features)} numerical and {len(categorical_features)} categorical features for LGBM.")

    print("Converting categorical features to 'category' dtype for LGBM...")
    category_map = {}
    X_train_lgbm, X_val_lgbm, X_test_lgbm = X_train.copy(), X_val.copy(), X_test.copy()
    active_categorical_features = []
    for col in categorical_features:
        if col in X_train_lgbm.columns:
            active_categorical_features.append(col)
            all_categories = pd.concat([
                X_train_lgbm[col].astype(str), X_val_lgbm[col].astype(str), X_test_lgbm[col].astype(str)
            ]).unique()
            cat_type = pd.CategoricalDtype(categories=all_categories, ordered=False)
            for df_part in [X_train_lgbm, X_val_lgbm, X_test_lgbm]:
                df_part[col] = df_part[col].astype(str).astype(cat_type)
            category_map[col] = cat_type
        else:
             print(f"Warning: Categorical feature '{col}' not found for LGBM.")

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
        else: scaler = None
    final_feature_names = active_numerical_features + active_categorical_features
    X_train_lgbm, X_val_lgbm, X_test_lgbm = X_train_lgbm[final_feature_names], X_val_lgbm[final_feature_names], X_test_lgbm[final_feature_names]

    print("Creating LGBM datasets...")
    lgb_train = lgb.Dataset(X_train_lgbm, label=y_train_encoded, categorical_feature=active_categorical_features or 'auto', feature_name=final_feature_names, free_raw_data=False)
    lgb_eval = lgb.Dataset(X_val_lgbm, label=y_val_encoded, reference=lgb_train, categorical_feature=active_categorical_features or 'auto', feature_name=final_feature_names, free_raw_data=False)

    sample_weight = None
    if class_weight_dict:
        print("Calculating sample weights for LGBM...")
        try:
             sample_weight = np.array([class_weight_dict.get(cls_idx, 1.0) for cls_idx in y_train_encoded])
             lgb_train.set_weight(sample_weight)
             print("Applied sample weights.")
        except Exception as e: print(f"Warning: Could not compute/apply sample weights for LGBM: {e}.")

    params = {
        'objective': 'multiclass', 'metric': ['multi_logloss', 'multi_error'],
        'num_class': num_classes, 'boosting_type': 'gbdt', 'n_estimators': 2500,
        'learning_rate': 0.02, 'num_leaves': 21, 'reg_alpha': 0.5, 'reg_lambda': 0.5,
        'colsample_bytree': 0.7, 'subsample': 0.7, 'min_child_samples': 50,
        'max_depth': -1, 'seed': 42, 'n_jobs': -1, 'verbose': -1,
    }

    print("Starting LGBM model training...")
    evals_result = {}
    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=True), lgb.log_evaluation(period=50), lgb.record_evaluation(evals_result)]
    lgbm_model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_eval], valid_names=['train', 'eval'], callbacks=callbacks)
    print("LGBM Training finished.")

    # Evaluate LightGBM Model
    print("\nEvaluating Final LGBM model on the test set...") # <-- MODIFIED PRINT
    y_pred_proba = lgbm_model.predict(X_test_lgbm, num_iteration=lgbm_model.best_iteration)
    y_pred_indices = np.argmax(y_pred_proba, axis=1) # Predicted class indices

    accuracy = accuracy_score(y_test_encoded, y_pred_indices)
    print(f"Final LGBM Test Accuracy: {accuracy:.4f}") # <-- MODIFIED PRINT

    # Calculate AUC (if multiclass, requires One-vs-Rest or similar strategy)
    # For multiclass, scikit-learn's roc_auc_score can handle it with 'ovr' or 'ovo'
    # Ensure y_test_encoded is 1D array of true labels, and y_pred_proba is 2D array of probabilities
    auc_score_val = np.nan # Default to NaN
    if num_classes > 1 : # AUC makes sense for more than 1 class
        try:
            from sklearn.metrics import roc_auc_score
            # If binary classification (num_classes == 2), use probabilities of the positive class
            if num_classes == 2:
                # Assuming the second class (index 1) is the positive class for AUC
                auc_score_val = roc_auc_score(y_test_encoded, y_pred_proba[:, 1])
            else: # Multiclass
                # 'ovr' (One-vs-Rest) is common for multiclass AUC
                # Ensure labels for roc_auc_score are the unique class indices present in y_test_encoded
                # and y_pred_proba columns align with these labels.
                # For 'ovr', it averages the AUC of each class against the rest.
                unique_test_labels = np.unique(y_test_encoded)
                if len(unique_test_labels) < 2:
                    print("Warning: AUC not calculated. Less than 2 classes in the test set after filtering.")
                elif len(unique_test_labels) < num_classes and num_classes > 2:
                    print(f"Warning: Only {len(unique_test_labels)} classes present in test set out of {num_classes} total. AUC might be misleading or error.")
                    # Option: Remap y_test_encoded and filter y_pred_proba to only include columns for present classes
                    # This can get complex if label_encoder.classes_ order doesn't match y_pred_proba columns for some reason.
                    # For simplicity, we'll try with the full y_pred_proba, roc_auc_score handles it if labels are provided.
                    auc_score_val = roc_auc_score(y_test_encoded, y_pred_proba, multi_class='ovr', average='weighted', labels=np.arange(num_classes))
                else:
                     auc_score_val = roc_auc_score(y_test_encoded, y_pred_proba, multi_class='ovr', average='weighted', labels=np.arange(num_classes))

            print(f"Final LGBM Test AUC: {auc_score_val:.4f}") # <-- MODIFIED PRINT
        except ValueError as e:
            print(f"Warning: Could not calculate AUC: {e}. Ensure all classes are present or handle appropriately.")
            print(f"  y_test_encoded shape: {y_test_encoded.shape}, unique labels: {np.unique(y_test_encoded)}")
            print(f"  y_pred_proba shape: {y_pred_proba.shape}")
        except ImportError:
            print("Warning: sklearn.metrics.roc_auc_score not found. AUC not calculated.")
    else:
        print("AUC not calculated (num_classes <= 1).")


    # Top-K Accuracy (already present, just ensure it's printed if calculated)
    try:
        top_k_acc_val = top_k_accuracy_score(y_test_encoded, y_pred_proba, k=3, labels=np.arange(num_classes))
        print(f"LGBM Test Top-3 Accuracy: {top_k_acc_val:.4f}") # Original print for Top-K
    except ValueError:
        print("Warning: Cannot calculate top-3 accuracy.")
        top_k_acc_val = np.nan


    # Classification Report
    print("\nFinal LGBM Classification Report (Test Set):") # <-- MODIFIED PRINT
    try:
        # Get class names from the label_encoder
        # Ensure label_encoder was passed correctly and is fitted
        if label_encoder and hasattr(label_encoder, 'classes_'):
            # Ensure target_names correspond to the unique sorted labels in y_test_encoded
            # or use indices if class names are too many / problematic
            report_labels = np.unique(y_test_encoded) # Use only labels present in test set for report
            target_names_for_report = label_encoder.inverse_transform(report_labels)

            # If there are many classes, the report can be very long.
            # Consider printing only for a subset or if num_classes is small.
            if num_classes > 25: # Example threshold
                print(f"(Report for {num_classes} classes might be too long, showing for {len(report_labels)} present in test)")
            
            # Check if all predicted indices are within the range of learned classes
            if np.max(y_pred_indices) >= num_classes or np.min(y_pred_indices) < 0:
                print("Warning: Predicted indices are out of bounds of num_classes. Clamping predictions for report.")
                y_pred_indices_clamped = np.clip(y_pred_indices, 0, num_classes - 1)
                class_report = classification_report(y_test_encoded, y_pred_indices_clamped, labels=report_labels, target_names=target_names_for_report, zero_division=0)
            else:
                class_report = classification_report(y_test_encoded, y_pred_indices, labels=report_labels, target_names=target_names_for_report, zero_division=0)
            
            print(class_report)
        else:
            print("Warning: Label encoder not available or not fitted. Printing report with numeric labels.")
            class_report = classification_report(y_test_encoded, y_pred_indices, zero_division=0)
            print(class_report)

    except ValueError as e:
        print(f"Warning: Could not generate classification report: {e}")
        print("This can happen if not all classes are present in y_test_encoded or y_pred_indices after some filtering.")
    except Exception as e:
        print(f"An unexpected error occurred during classification report generation: {e}")

    model_save_path = f'switch_lgbm_model_v1_{label_suffix}.txt'
    print(f"Saving LGBM model to {model_save_path}")
    lgbm_model.save_model(model_save_path)

    lgbm_info_path = f'switch_lgbm_feature_info_v1_{label_suffix}.joblib'
    print(f"Saving LGBM feature info to {lgbm_info_path}")
    joblib.dump({'numerical_features': active_numerical_features, 'categorical_features': active_categorical_features,
                 'feature_names_in_order': final_feature_names, 'category_map': category_map}, lgbm_info_path)
    if scaler and active_numerical_features:
        scaler_path = f'switch_lgbm_scaler_v1_{label_suffix}.joblib'
        joblib.dump(scaler, scaler_path)
        print(f"LGBM scaler saved to {scaler_path}")

    return lgbm_model

# --- Main execution function (MODIFIED for switch prediction) ---
def run_switch_training(parquet_path, model_type='tensorflow', feature_set='full',
                        min_turn=0, test_split_size=0.2, val_split_size=0.15,
                        epochs=30, batch_size=256, learning_rate=0.001):
    print(f"--- Starting P2 Switch Predictor Training (V1) ---")
    print(f"Model type: {model_type.upper()}")
    print(f"Feature Set: {feature_set.upper()}")
    print(f"Loading data from: {parquet_path}")
    try:
        df = pd.read_parquet(parquet_path)
        print(f"Original data shape: {df.shape}")
    except Exception as e:
        print(f"Error loading Parquet file: {e}"); return

    print("\nFiltering data...")
    original_rows = len(df)
    df = df.dropna(subset=['action_taken']) # Essential for target
    print(f"Rows after dropping NaN action_taken: {len(df)}")

    # --- Key Filter: Player 2 switch actions ---
    print("Filtering for Player 2 switch actions...")
    df = df[
        (df['player_to_move'] == 'p2') &
        (df['action_taken'].astype(str).str.startswith('switch:'))
    ].copy()
    if df.empty:
        print("Error: No 'switch:' actions found for player p2. Cannot train.")
        return
    print(f"Rows after filtering for P2 switch actions: {len(df)} (Removed {original_rows - len(df)})")

    # --- Target Variable: Switched-to Species ---
    y_raw = df['action_taken'].str.replace('switch:', '', regex=False).str.strip()
    # Remove rows if species name became empty after stripping (e.g. action was just "switch:")
    valid_y_indices = y_raw[y_raw != ''].index
    y_raw = y_raw[y_raw.index.isin(valid_y_indices)]
    df = df[df.index.isin(valid_y_indices)]

    if df.empty or y_raw.empty:
        print("Error: No valid data remaining after P2 switch filtering and target extraction.")
        return
    print(f"Target variable 'y_raw' (switched-to species for P2) created. Example: {y_raw.head().tolist()}")


    X = None
    numerical_features = []
    categorical_features = []
    label_encoder_suffix = f"{feature_set}_p2switch" # Unique suffix for switch predictor

    if feature_set == 'simplified':
        print("\n--- Using ENHANCED SIMPLIFIED feature set for P2 switch ---") # Updated print message
        
        temp_numerical_features = []
        temp_categorical_features = []

        # 1. Extract P1's (Opponent) and P2's (Self, switching out) Active Pokémon Details
        print("  Extracting P1's and P2's active Pokémon details...")
        active_pokemon_data_for_rows = {} # Store all active data for P1 and P2 per row
        p2_team_tera_status_for_rows = {} # Store if P2 has already used Tera on any team member

        for i_row, (idx, row) in enumerate(df.iterrows()): # df is already filtered for P2 switches
            active_p1_slot_num = -1
            active_p2_slot_num = -1 # For the Pokemon P2 is switching OUT

            for i_slot in range(1, 7):
                if row.get(f'p1_slot{i_slot}_is_active', 0) == 1:
                    active_p1_slot_num = i_slot
                if row.get(f'p2_slot{i_slot}_is_active', 0) == 1: # Pokemon P2 is switching out
                    active_p2_slot_num = i_slot
            
            current_row_data = {}
            # P1's Active (Opponent)
            if active_p1_slot_num != -1:
                current_row_data['p1_active_species'] = row.get(f'p1_slot{active_p1_slot_num}_species', 'Unknown')
                current_row_data['p1_active_hp_perc'] = row.get(f'p1_slot{active_p1_slot_num}_hp_perc', 100)
                current_row_data['p1_active_status'] = row.get(f'p1_slot{active_p1_slot_num}_status', 'none')
                current_row_data['p1_active_terastallized'] = row.get(f'p1_slot{active_p1_slot_num}_terastallized', 0)
                current_row_data['p1_active_tera_type'] = row.get(f'p1_slot{active_p1_slot_num}_tera_type', 'none')
            else: 
                current_row_data['p1_active_species'] = 'Unknown'
                current_row_data['p1_active_hp_perc'] = 100
                current_row_data['p1_active_status'] = 'none'
                current_row_data['p1_active_terastallized'] = 0
                current_row_data['p1_active_tera_type'] = 'none'

            # P2's Active (Pokemon being switched OUT by P2)
            if active_p2_slot_num != -1:
                current_row_data['p2_current_active_species'] = row.get(f'p2_slot{active_p2_slot_num}_species', 'Unknown')
                current_row_data['p2_current_active_hp_perc'] = row.get(f'p2_slot{active_p2_slot_num}_hp_perc', 100)
                current_row_data['p2_current_active_status'] = row.get(f'p2_slot{active_p2_slot_num}_status', 'none')
                # p2_current_active_terastallized is implicitly handled by p2_can_terastallize later
            else:
                current_row_data['p2_current_active_species'] = 'Unknown'
                current_row_data['p2_current_active_hp_perc'] = 100
                current_row_data['p2_current_active_status'] = 'none'
            
            active_pokemon_data_for_rows[idx] = current_row_data

            # Check if P2 has already used Tera on any team member for p2_can_terastallize
            p2_has_used_tera_on_team = 0
            for i_slot_tera_check in range(1, 7):
                if row.get(f'p2_slot{i_slot_tera_check}_terastallized', 0) == 1:
                    p2_has_used_tera_on_team = 1
                    break
            p2_team_tera_status_for_rows[idx] = {'p2_can_terastallize': 1 - p2_has_used_tera_on_team}

        
        active_pokemon_df = pd.DataFrame.from_dict(active_pokemon_data_for_rows, orient='index')
        p2_can_tera_df = pd.DataFrame.from_dict(p2_team_tera_status_for_rows, orient='index')

        # Define features from active_pokemon_df and their types
        active_features_to_use = []
        if 'p1_active_species' in active_pokemon_df.columns: temp_categorical_features.append('p1_active_species'); active_features_to_use.append('p1_active_species')
        if 'p1_active_hp_perc' in active_pokemon_df.columns: temp_numerical_features.append('p1_active_hp_perc'); active_features_to_use.append('p1_active_hp_perc')
        if 'p1_active_status' in active_pokemon_df.columns: temp_categorical_features.append('p1_active_status'); active_features_to_use.append('p1_active_status')
        if 'p1_active_terastallized' in active_pokemon_df.columns: temp_numerical_features.append('p1_active_terastallized'); active_features_to_use.append('p1_active_terastallized')
        if 'p1_active_tera_type' in active_pokemon_df.columns: temp_categorical_features.append('p1_active_tera_type'); active_features_to_use.append('p1_active_tera_type')
        
        if 'p2_current_active_species' in active_pokemon_df.columns: temp_categorical_features.append('p2_current_active_species'); active_features_to_use.append('p2_current_active_species')
        if 'p2_current_active_hp_perc' in active_pokemon_df.columns: temp_numerical_features.append('p2_current_active_hp_perc'); active_features_to_use.append('p2_current_active_hp_perc')
        if 'p2_current_active_status' in active_pokemon_df.columns: temp_categorical_features.append('p2_current_active_status'); active_features_to_use.append('p2_current_active_status')

        features_df_list_for_X_concat = []
        if active_features_to_use: # Check if any active features were actually added
            features_df_list_for_X_concat.append(active_pokemon_df[active_features_to_use])


        # 2. Player 2's (Bot's) Entire Team Details (excluding the one being switched out, if identified)
        print("  Adding P2's full team details (potential switch-ins)...")
        p2_team_cols_to_add_to_X = []
        for i in range(1, 7): # Slots 1 to 6 for P2
            slot_prefix = f"p2_slot{i}"
            features_to_check = {
                f"{slot_prefix}_species": temp_categorical_features,
                f"{slot_prefix}_hp_perc": temp_numerical_features,
                f"{slot_prefix}_status": temp_categorical_features,
                f"{slot_prefix}_is_fainted": temp_numerical_features,
                f"{slot_prefix}_terastallized": temp_numerical_features # Has this specific Pokemon terastallized?
            }
            for feature_col, type_list in features_to_check.items():
                if feature_col in df.columns:
                    type_list.append(feature_col)
                    p2_team_cols_to_add_to_X.append(feature_col)
        
        if p2_team_cols_to_add_to_X:
            features_df_list_for_X_concat.append(df[p2_team_cols_to_add_to_X])

        # Add p2_can_terastallize (team-wide flag)
        if 'p2_can_terastallize' in p2_can_tera_df.columns:
            temp_numerical_features.append('p2_can_terastallize')
            features_df_list_for_X_concat.append(p2_can_tera_df[['p2_can_terastallize']])

        # 3. Field, Hazard, and General Context Features
        print("  Adding field, hazard, and general context features...")
        context_field_hazard_cols_to_add_to_X = []
        
        # Field Conditions
        if 'field_weather' in df.columns: temp_categorical_features.append('field_weather'); context_field_hazard_cols_to_add_to_X.append('field_weather')
        if 'field_terrain' in df.columns: temp_categorical_features.append('field_terrain'); context_field_hazard_cols_to_add_to_X.append('field_terrain')
        
        # Hazards (P1's side - affecting P2)
        for hazard in ['stealth_rock', 'spikes', 'toxic_spikes', 'sticky_web']:
            col_name = f'p1_hazard_{hazard}'
            if col_name in df.columns: temp_numerical_features.append(col_name); context_field_hazard_cols_to_add_to_X.append(col_name)
        # Hazards (P2's side - affecting P1, but good context)
        for hazard in ['stealth_rock', 'spikes', 'toxic_spikes', 'sticky_web']:
            col_name = f'p2_hazard_{hazard}'
            if col_name in df.columns: temp_numerical_features.append(col_name); context_field_hazard_cols_to_add_to_X.append(col_name)

        # General Context
        if 'last_move_p1' in df.columns: temp_categorical_features.append('last_move_p1'); context_field_hazard_cols_to_add_to_X.append('last_move_p1')
        if 'last_move_p2' in df.columns: temp_categorical_features.append('last_move_p2'); context_field_hazard_cols_to_add_to_X.append('last_move_p2')
        # turn_number is already available directly in df, will be added if selected for X
        if 'turn_number' in df.columns: temp_numerical_features.append('turn_number'); context_field_hazard_cols_to_add_to_X.append('turn_number')
        
        if context_field_hazard_cols_to_add_to_X:
            features_df_list_for_X_concat.append(df[context_field_hazard_cols_to_add_to_X])

        # Concatenate all selected features into X
        if features_df_list_for_X_concat:
            X = pd.concat(features_df_list_for_X_concat, axis=1)
        else:
            X = pd.DataFrame(index=df.index) 
            print("Warning: No features selected for 'simplified' set. X is empty.")

        # Finalize feature lists (remove duplicates and ensure order if necessary)
        numerical_features = sorted(list(set(temp_numerical_features)))
        categorical_features = sorted(list(set(temp_categorical_features)))
        
        overlap = set(numerical_features) & set(categorical_features)
        if overlap:
            print(f"Warning: Overlap detected in simplified features: {overlap}. Removing from numerical.")
            numerical_features = [f for f in numerical_features if f not in overlap]

        print(f"Enhanced Simplified X shape: {X.shape if X is not None else 'X is None'}")
        if X is not None and not X.empty:
            print(f"  Numerical features ({len(numerical_features)}): {numerical_features[:15]}... (first 15 shown)") # Show sample
            print(f"  Categorical features ({len(categorical_features)}): {categorical_features[:15]}... (first 15 shown)") # Show sample
        
        del active_pokemon_df, p2_can_tera_df, features_df_list_for_X_concat, active_pokemon_data_for_rows, p2_team_tera_status_for_rows
        gc.collect()

    print(f"\nFinal NaN Check for '{feature_set}' set...")
    # (Simplified NaN check from action predictor)
    if X.isnull().sum().sum() > 0:
        print("Warning: NaNs still present before final encoding. Attempting fill.")
        for col in X.columns:
            if X[col].isnull().any():
                if pd.api.types.is_numeric_dtype(X[col]): X[col].fillna(0, inplace=True)
                else:
                    if pd.api.types.is_categorical_dtype(X[col]):
                        if 'Unknown' not in X[col].cat.categories: X[col] = X[col].cat.add_categories(['Unknown'])
                        X[col].fillna('Unknown', inplace=True)
                    else: X[col].fillna('Unknown', inplace=True) # Convert to object then category later if needed
    if X.isnull().sum().sum() > 0:
        print(f"Error: NaNs still present AFTER final handling: {X.isnull().sum()[X.isnull().sum() > 0]}"); return

    print(f"\nFinal feature counts for '{feature_set}' set:")
    print(f"  Numerical: {len(numerical_features)}, Categorical: {len(categorical_features)}, Total in X: {X.shape[1]}")

    print(f"\nEncoding target variable (switched-to species)...")
    label_encoder = LabelEncoder()
    try:
        y_encoded = label_encoder.fit_transform(y_raw.astype(str))
    except Exception as e: print(f"Error encoding target: {e}."); return
    num_classes = len(label_encoder.classes_)
    print(f"Found {num_classes} unique Pokémon species P2 switched to.")
    if num_classes < 2: print("Error: Need at least 2 unique target species."); return

    label_encoder_path = f'switch_label_encoder_v1_{label_encoder_suffix}.joblib'
    joblib.dump(label_encoder, label_encoder_path)
    print(f"Label encoder saved to {label_encoder_path}")
    del y_raw; gc.collect()

    print("\nSplitting data...")
    # (Data splitting logic - same as action predictor, using y_encoded for stratification if possible)
    try:
        stratify_option = y_encoded if num_classes < 1000 and np.min(np.bincount(y_encoded)) >= 2 else None
        X_train_full, X_test, y_train_full_encoded, y_test_encoded = train_test_split(
            X, y_encoded, test_size=test_split_size, random_state=42, stratify=stratify_option
        )
        val_size_relative = val_split_size / (1.0 - test_split_size) if (1.0 - test_split_size) > 0 else 0.15
        if not (0 < val_size_relative < 1): val_size_relative = 0.15
        
        stratify_option_2 = y_train_full_encoded if len(y_train_full_encoded) > 0 and np.min(np.bincount(y_train_full_encoded)) >=2 else None
        X_train, X_val, y_train_encoded, y_val_encoded = train_test_split(
            X_train_full, y_train_full_encoded, test_size=val_size_relative, random_state=42, stratify=stratify_option_2
        )
        if stratify_option is None or stratify_option_2 is None: print("Splitting without full stratification due to class imbalance or size.")
    except ValueError as e:
        print(f"Error during stratified splitting: {e}. Splitting randomly.")
        X_train_full, X_test, y_train_full_encoded, y_test_encoded = train_test_split(X, y_encoded, test_size=test_split_size, random_state=42)
        val_size_relative = val_split_size / (1.0 - test_split_size) if (1.0 - test_split_size) > 0 else 0.15
        X_train, X_val, y_train_encoded, y_val_encoded = train_test_split(X_train_full, y_train_full_encoded, test_size=val_size_relative, random_state=42)

    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")
    del X, df, X_train_full, y_train_full_encoded; gc.collect()

    print("\nCalculating class weights...")
    # (Class weight calculation - same as action predictor)
    if len(y_train_encoded) == 0: print("Error: y_train_encoded empty."); return
    unique_train_classes, _ = np.unique(y_train_encoded, return_counts=True)
    class_weight_dict = {cls_idx: 1.0 for cls_idx in range(num_classes)} # Default
    if len(unique_train_classes) >= 2:
        try:
             class_weights_values = compute_class_weight('balanced', classes=unique_train_classes, y=y_train_encoded)
             temp_dict = dict(zip(unique_train_classes, class_weights_values))
             for cls_idx in range(num_classes): # Ensure all classes have a weight
                 class_weight_dict[cls_idx] = temp_dict.get(cls_idx, 1.0)
        except Exception as e: print(f"Error calculating class weights: {e}. Using uniform.")


    X_train_processed, X_val_processed, X_test_processed = None, None, None
    preprocessor = None
    feature_lists_path = f'switch_feature_lists_v1_{label_encoder_suffix}.joblib'
    preprocessor_path = f'switch_tf_preprocessor_v1_{label_encoder_suffix}.joblib'

    final_train_cols = X_train.columns.tolist()
    numerical_features = [f for f in numerical_features if f in final_train_cols] # Ensure consistency
    categorical_features = [f for f in categorical_features if f in final_train_cols]
    joblib.dump({'feature_columns_final': final_train_cols, 'numerical_features': numerical_features, 'categorical_features': categorical_features}, feature_lists_path)

    if model_type == 'tensorflow':
        print("\nSetting up TF preprocessing pipeline...")
        # (TF preprocessing - same as action predictor)
        transformers = []
        if numerical_features: transformers.append(('num', Pipeline(steps=[('scaler', StandardScaler())]), numerical_features))
        if categorical_features: transformers.append(('cat', Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))]), categorical_features))
        if not transformers: print("Error: No transformers for TF!"); return
        preprocessor = ColumnTransformer(transformers=transformers, remainder='drop', sparse_threshold=0.3)
        try:
            X_train_processed = preprocessor.fit_transform(X_train)
            X_val_processed = preprocessor.transform(X_val)
            X_test_processed = preprocessor.transform(X_test)
            joblib.dump(preprocessor, preprocessor_path)
            print(f"TF preprocessor saved. Processed shapes - Train: {X_train_processed.shape}")
        except Exception as e: print(f"Error during TF preprocessing: {e}"); return
        del X_train, X_val, X_test; gc.collect()
    elif model_type == 'lightgbm':
        X_train_processed, X_val_processed, X_test_processed = X_train, X_val, X_test
    else:
        print(f"Error: Unknown model_type '{model_type}'"); return

    print(f"\n--- Initiating {model_type.upper()} Switch Model Training ({feature_set.upper()} features) ---")
    if model_type == 'tensorflow':
         if X_train_processed is not None:
             train_tensorflow_switch_predictor(X_train_processed, X_val_processed, X_test_processed,
                                               y_train_encoded, y_val_encoded, y_test_encoded,
                                               num_classes, class_weight_dict, label_encoder,
                                               epochs, batch_size, learning_rate,
                                               label_suffix=label_encoder_suffix)
         else: print("Skipping TF training due to preprocessing errors.")
    elif model_type == 'lightgbm':
         train_lgbm_switch_predictor(X_train_processed, X_val_processed, X_test_processed,
                                     y_train_encoded, y_val_encoded, y_test_encoded,
                                     numerical_features, categorical_features, # Pass filtered lists
                                     num_classes, class_weight_dict, label_encoder,
                                     label_suffix=label_encoder_suffix)

# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train P2 switch predictor (V1).")
    parser.add_argument("parquet_file", type=str, help="Path to the input Parquet file.")
    parser.add_argument("--model_type", choices=['tensorflow', 'lightgbm'], default='lightgbm', help="Type of model to train.")
    parser.add_argument("--feature_set", choices=['full', 'medium', 'simplified'], default='full', help="Feature set to use.")
    parser.add_argument("--min_turn", type=int, default=1, help="Minimum turn number to include.")
    parser.add_argument("--test_split", type=float, default=0.2, help="Fraction for test set.")
    parser.add_argument("--val_split", type=float, default=0.15, help="Fraction for validation set (relative to train_full).")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs (TF only).")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size (TF only).")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (TF only).")
    args = parser.parse_args()

    if args.test_split + args.val_split >= 1.0 or args.test_split <= 0 or args.val_split <= 0:
        print("Error: Invalid split sizes. test_split + val_split must be < 1.0, and both > 0."); exit(1)

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