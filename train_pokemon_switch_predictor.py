import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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
import optuna
import json
from collections import Counter

# Suppress TensorFlow/warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
USAGE_STATS_JSON = "gen9ou-0.json"

def get_switch_slot(row, target_species):
    for slot in range(1, 7):
        if row.get(f'p1_slot{slot}_is_active', 0) == 1:
            continue  # Skip active slot
        if row.get(f'p1_slot{slot}_species', '').lower() == target_species.lower():
            return slot - 1 if slot > row['p1_active_slot'] else slot  # Relative bench slot (1-5), adjust as needed
    return None  # If no match (rare), drop row

def load_smogon_moves(json_filepath):
    """Loads Smogon usage stats JSON and extracts valid moves for each Pokemon."""
    print(f"Loading Smogon usage stats from: {json_filepath}")
    pokemon_valid_moves = {}
    metagame = None

    try:
        if not os.path.exists(json_filepath):
             raise FileNotFoundError(f"Smogon JSON file not found at '{json_filepath}'")
        with open(json_filepath, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # Extract Metagame Info
        if 'info' in raw_data and isinstance(raw_data['info'], dict):
            metagame = raw_data['info'].get('metagame')
            if metagame: print(f"  Detected Metagame: {metagame}")
            else: print("  Warning: Metagame info not found in JSON['info'].")
        else: print("  Warning: 'info' key not found or not a dict in JSON.")

        # Extract Move Data
        if 'data' not in raw_data or not isinstance(raw_data['data'], dict):
            print("  Error: 'data' key not found or is not a dictionary in JSON. Cannot extract moves.")
            return {}, metagame

        smogon_data = raw_data['data']
        usage_dict = {pokemon_name: pokemon_data.get('usage', 0) for pokemon_name, pokemon_data in smogon_data.items()}
        top_100 = sorted(usage_dict.items(), key=lambda x: x[1], reverse=True)[:100]
        top_100_names = [name.lower().replace(' ', '').replace('-', '') for name, _ in top_100]
        pokemon_count = 0
        move_count_total = 0

        for pokemon_name, pokemon_data in smogon_data.items():
            # Standardize Pokemon Name from Smogon data (lowercase, no spaces/hyphens)
            # This should match how poke-env species IDs are likely formatted
            pokemon_key = pokemon_name.lower().replace(' ', '').replace('-','')

            if isinstance(pokemon_data, dict) and 'Moves' in pokemon_data and isinstance(pokemon_data['Moves'], dict):
                valid_moves_set = set()
                for move_key in pokemon_data['Moves'].keys():
                    # Format should match label_encoder format ("move:moveid")
                    standardized_action = f"move:{move_key.lower()}" # Assuming encoder used lowercase move IDs
                    valid_moves_set.add(standardized_action)

                if valid_moves_set:
                    pokemon_valid_moves[pokemon_key] = valid_moves_set # Use standardized key
                    pokemon_count += 1
                    move_count_total += len(valid_moves_set)

        print(f"  Successfully extracted moves for {pokemon_count} Pokemon.")
        print(f"  Total unique Pokemon-Move combinations found: {move_count_total}")
        return pokemon_valid_moves, metagame, top_100_names  

    except FileNotFoundError:
        print(f"  Error: Smogon JSON file not found at '{json_filepath}'")
        raise
    except json.JSONDecodeError as e:
        print(f"  Error: Failed to decode Smogon JSON file '{json_filepath}'. Invalid JSON: {e}")
        raise
    except Exception as e:
        print(f"  Error: An unexpected error occurred while loading Smogon JSON: {e}")
        raise

# --- TF Training Function ---
def train_tensorflow_switch_target_predictor(X_train_processed, X_val_processed, X_test_processed,
                                             y_train, y_val, y_test,
                                             class_weight_dict,
                                             num_classes, 
                                             epochs=20, batch_size=128, learning_rate=0.001,
                                             model_suffix=""):
    print(f"\n--- Training TensorFlow Switch Target Predictor (Multi-Class) ---")
    print(f"Input shape: {X_train_processed.shape[1]}")
    print(f"Target is multi-class with {num_classes} possible classes (Pokémon).")

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
        # *** Output layer for multi-class classification ***
        Dense(num_classes, activation='softmax')
    ])

    optimizer = Adam(learning_rate=learning_rate)
    # ***  Loss function for integer-based multi-class labels ***
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # Train the Model
    print("\nStarting TF model training...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    history = model.fit(
        X_train_processed, y_train,
        validation_data=(X_val_processed, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
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

    # Save Model
    model_save_path = f'switch_target_predictor_tf_model_{model_suffix}.keras'
    print(f"Saving TF model to {model_save_path}")
    try:
        model.save(model_save_path)
        print("TF Model saved.")
    except Exception as e:
        print(f"Error saving TF model: {e}")

    return history, model


# --- LGBM Training Function ---
def train_lgbm_switch_target_predictor(X_train, X_val, X_test,
                                       y_train, y_val, y_test,
                                       numerical_features, categorical_features,
                                       class_weight_dict,
                                       num_classes, label_encoder, 
                                       model_suffix="",
                                       perform_hpo=True,
                                       n_hpo_trials=50):
    print(f"\n--- Training LightGBM Switch Target Predictor (Multi-Class) ---")
    print(f"Using {len(numerical_features)} numerical and {len(categorical_features)} categorical features for LGBM.")
    print(f"Perform Hyperparameter Optimization (Optuna): {perform_hpo}")

    def objective(trial, current_X_train, current_y_train, current_X_val, current_y_val,
                  numerical_features_obj, categorical_features_obj, class_weight_dict_obj, num_classes_obj):
        X_train_trial = current_X_train.copy()
        X_val_trial = current_X_val.copy()

        trial_active_categorical_features = []
        for col in categorical_features_obj:
            if col in X_train_trial.columns:
                trial_active_categorical_features.append(col)
                all_categories = pd.concat([
                    X_train_trial[col].astype(str).fillna('Unknown'),
                    X_val_trial[col].astype(str).fillna('Unknown')
                ]).unique()
                cat_type = pd.CategoricalDtype(categories=sorted([c for c in all_categories if isinstance(c, str)]), ordered=False)
                try:
                    X_train_trial[col] = X_train_trial[col].astype(str).fillna('Unknown').astype(cat_type)
                    X_val_trial[col] = X_val_trial[col].astype(str).fillna('Unknown').astype(cat_type)
                except ValueError:
                    X_train_trial[col] = X_train_trial[col].astype(str).fillna('Unknown').astype('category')
                    X_val_trial[col] = X_val_trial[col].astype(str).fillna('Unknown').astype('category')

        trial_active_numerical_features = [f for f in numerical_features_obj if f in X_train_trial.columns]
        if trial_active_numerical_features:
            scaler_trial = StandardScaler()
            try:
                X_train_trial[trial_active_numerical_features] = scaler_trial.fit_transform(X_train_trial[trial_active_numerical_features])
                X_val_trial[trial_active_numerical_features] = scaler_trial.transform(X_val_trial[trial_active_numerical_features])
            except ValueError:
                pass

        trial_final_feature_names = trial_active_numerical_features + trial_active_categorical_features
        X_train_trial_processed = X_train_trial[trial_final_feature_names]
        X_val_trial_processed = X_val_trial[trial_final_feature_names]

        lgb_train_trial = lgb.Dataset(X_train_trial_processed, label=current_y_train,
                                      categorical_feature=[f for f in trial_active_categorical_features if f in trial_final_feature_names] or 'auto',
                                      free_raw_data=False)
        lgb_eval_trial = lgb.Dataset(X_val_trial_processed, label=current_y_val, reference=lgb_train_trial,
                                     categorical_feature=[f for f in trial_active_categorical_features if f in trial_final_feature_names] or 'auto',
                                     free_raw_data=False)

        if class_weight_dict_obj:
            sample_weight_trial = current_y_train.map(class_weight_dict_obj).fillna(1.0).values
            lgb_train_trial.set_weight(sample_weight_trial)

        # *** Search space and objective for multi-class ***
        params = {
            'objective': 'multiclass',
            'metric': 'multi_logloss', # Optimize for logloss
            'num_class': num_classes_obj, # Specify number of classes
            'boosting_type': 'gbdt', 'verbosity': -1, 'n_jobs': -1, 'seed': 42,
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }

        model = lgb.train(params, lgb_train_trial, valid_sets=[lgb_eval_trial],valid_names=['eval'],
                          callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])

        # Return the validation loss for minimization
        return model.best_score['eval']['multi_logloss']

    X_train_lgbm = X_train.copy()
    X_val_lgbm = X_val.copy()
    X_test_lgbm = X_test.copy()

    best_params_from_hpo = {}
    if perform_hpo:
        print(f"\n--- Starting Optuna Hyperparameter Optimization ({n_hpo_trials} trials) ---")
        # *** Study direction is 'minimize' for logloss ***
        study = optuna.create_study(direction='minimize', study_name="lgbm_switch_target_predictor_tuning")
        study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val,
                                               numerical_features, categorical_features, class_weight_dict, num_classes),
                       n_trials=n_hpo_trials)

        print("\n--- Optuna HPO Finished ---")
        best_trial = study.best_trial
        print(f"  Value (LogLoss): {best_trial.value:.4f}")
        print("  Params: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
        best_params_from_hpo = best_trial.params
    else:
        print("Skipping Hyperparameter Optimization.")

    # --- Final Model Training  ---
    print("\n--- Preparing Data for Final Model Training ---")
    category_map = {}
    active_categorical_features = []
    for col in categorical_features:
        if col in X_train_lgbm.columns:
            active_categorical_features.append(col)
            all_categories = pd.concat([
                X_train_lgbm[col].astype(str).fillna('Unknown'),
                X_val_lgbm[col].astype(str).fillna('Unknown'),
                X_test_lgbm[col].astype(str).fillna('Unknown')
            ]).unique()
            cat_type = pd.CategoricalDtype(categories=sorted([c for c in all_categories if isinstance(c, str)]), ordered=False)
            try:
                X_train_lgbm[col] = X_train_lgbm[col].astype(str).fillna('Unknown').astype(cat_type)
                X_val_lgbm[col] = X_val_lgbm[col].astype(str).fillna('Unknown').astype(cat_type)
                X_test_lgbm[col] = X_test_lgbm[col].astype(str).fillna('Unknown').astype(cat_type)
                category_map[col] = cat_type
            except ValueError:
                X_train_lgbm[col] = X_train_lgbm[col].astype(str).fillna('Unknown').astype('category')
                X_val_lgbm[col] = X_val_lgbm[col].astype(str).fillna('Unknown').astype('category')
                X_test_lgbm[col] = X_test_lgbm[col].astype(str).fillna('Unknown').astype('category')
                category_map[col] = X_train_lgbm[col].dtype

    active_numerical_features = [f for f in numerical_features if f in X_train_lgbm.columns]
    scaler = None
    features_scaled = []
    if active_numerical_features:
        print("Scaling numerical features for final model...")
        scaler = StandardScaler()
        try:
            X_train_lgbm[active_numerical_features] = scaler.fit_transform(X_train_lgbm[active_numerical_features])
            X_val_lgbm[active_numerical_features] = scaler.transform(X_val_lgbm[active_numerical_features])
            X_test_lgbm[active_numerical_features] = scaler.transform(X_test_lgbm[active_numerical_features])
            features_scaled = active_numerical_features
            print("Numerical scaling complete.")
        except ValueError as e:
            print(f"Warning: ValueError during final scaling: {e}. Skipping scaling.")
            scaler = None; features_scaled = []

    final_feature_names = active_numerical_features + active_categorical_features
    final_feature_names = [col for col in final_feature_names if col in X_train_lgbm.columns]
    X_train_lgbm_final = X_train_lgbm[final_feature_names]
    X_val_lgbm_final = X_val_lgbm[final_feature_names]
    X_test_lgbm_final = X_test_lgbm[final_feature_names]

    lgb_train_final = lgb.Dataset(X_train_lgbm_final, label=y_train,
                                  categorical_feature=[f for f in active_categorical_features if f in final_feature_names] or 'auto',
                                  feature_name=final_feature_names, free_raw_data=False)
    lgb_eval_final = lgb.Dataset(X_val_lgbm_final, label=y_val, reference=lgb_train_final,
                                 categorical_feature=[f for f in active_categorical_features if f in final_feature_names] or 'auto',
                                 feature_name=final_feature_names, free_raw_data=False)
    if class_weight_dict:
        sample_weight_final = y_train.map(class_weight_dict).fillna(1.0).values
        lgb_train_final.set_weight(sample_weight_final)

    # ***  Final LGBM parameters for multi-class ***
    final_params = {
        'objective': 'multiclass',
        'metric': ['multi_logloss', 'multi_error'],
        'num_class': num_classes,
        'boosting_type': 'gbdt', 'seed': 42, 'n_jobs': -1, 'verbose': -1,
        'learning_rate': 0.006280820599565553, 'n_estimators': 2109, 'num_leaves': 88,
        'reg_alpha': 0.000002817854459142025, 'reg_lambda': 0.0031492970379127117, 'colsample_bytree': 0.667840523849307,
        'subsample': 0.7587846873233649, 'min_child_samples': 41,
    }
    final_params.update(best_params_from_hpo)

    print(f"\nStarting Final LGBM model training with parameters: {final_params}")
    lgbm_model = lgb.train(final_params, lgb_train_final,
                           valid_sets=[lgb_train_final, lgb_eval_final],
                           valid_names=['train', 'eval'],
                           callbacks=[lgb.early_stopping(100, verbose=True), lgb.log_evaluation(50)])
    print("Final LGBM Training finished.")
    importances = pd.DataFrame({'feature': final_feature_names, 'importance': lgbm_model.feature_importance()})
    importances = importances.sort_values('importance', ascending=False)
    print(importances.head(50))  # Log top features
    joblib.dump(importances, f'switch_target_feature_importances_{model_suffix}.joblib')

    # Evaluate Final LightGBM Model
    print("\nEvaluating Final LGBM model on the test set...")
    try:
        y_pred_proba = lgbm_model.predict(X_test_lgbm_final, num_iteration=lgbm_model.best_iteration)
        y_pred_class = np.argmax(y_pred_proba, axis=1) # Get the class with the highest probability
        accuracy = accuracy_score(y_test, y_pred_class)
        print(f"Final LGBM Test Accuracy: {accuracy:.4f}")

        print("\nFinal LGBM Classification Report (Test Set):")
        # ***  Use label_encoder to show Pokémon names in the report ***
        # Filter target names to only include classes present in the test set to avoid errors
        unique_test_labels = np.unique(y_test)
        filtered_target_names = [f"Slot {label_encoder.classes_[i]}" for i in unique_test_labels]  # e.g., "Slot 0", "Slot 1"
        print(classification_report(y_test, y_pred_class, labels=unique_test_labels, target_names=filtered_target_names))

    except Exception as e:
        print(f"Error during final LGBM evaluation: {e}")

    #  New save paths for switch target prediction
    model_save_path = f'switch_target_predictor_lgbm_model_{model_suffix}.txt'
    lgbm_info_path = f'switch_target_predictor_lgbm_feature_info_{model_suffix}.joblib'
    scaler_path = f'switch_target_predictor_lgbm_scaler_{model_suffix}.joblib'

    print(f"Saving final LGBM model to {model_save_path}")
    lgbm_model.save_model(model_save_path)
    lgbm_info = {
        'numerical_features': active_numerical_features, 'categorical_features': active_categorical_features,
        'feature_names_in_order': final_feature_names, 'features_scaled': features_scaled,
        'category_map': category_map, 'best_hpo_params': best_params_from_hpo if perform_hpo else "HPO not performed"
    }
    joblib.dump(lgbm_info, lgbm_info_path)
    if scaler and features_scaled:
        joblib.dump(scaler, scaler_path)
    print(f"Final LGBM model and feature info saved.")

    return lgbm_model


def run_switch_target_training(parquet_path, model_type='tensorflow', feature_set='full',
                               min_turn=0, test_split_size=0.2, val_split_size=0.15,
                               epochs=30, batch_size=256, learning_rate=0.001):
    """Loads data, filters for switches, creates multi-class target, and trains a switch target predictor."""

    print(f"--- Starting Switch Target Predictor Training (Multi-Class) ---")
    print(f"Model type: {model_type.upper()}")
    print(f"Feature Set: {feature_set.upper()}")
    print(f"Loading data from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"Original data shape: {df.shape}")

    # The processing script should already do this implicitly, but it's safer to be explicit.
    df = df.sort_values(by=['replay_id', 'turn_number'], ascending=True).reset_index(drop=True)

    # Step 2: Identify rows that are valid preceding states.
    # A row is a valid preceding state if the *next* row belongs to the same replay.
    df['is_same_replay_as_next'] = (df['replay_id'] == df['replay_id'].shift(-1))

    # Step 3: Identify the actions that occurred in the *next* row.
    df['next_action'] = df['action_taken'].shift(-1)

    # Step 4: Find the rows where a switch happens in the *next* turn within the same replay.
    # These rows are our feature sets (X).
    is_pre_switch_state = df['next_action'].str.startswith('switch', na=False) & df['is_same_replay_as_next']
    
    # Step 5: Create the features (X) and targets (y).
    # X is the current row if the next action is a switch.
    X = df[is_pre_switch_state].copy()

    # y is the 'next_action' from those same rows, which is the switch action itself.
    y_labels = X['next_action']

    # Clean up columns that we created and that could leak information about the future.
    X = X.drop(columns=['is_same_replay_as_next', 'next_action'])

    print(f"Found {len(X)} valid pre-switch states to use for training.")
    if X.empty:
        print("Error: Could not construct a training set. No valid switch actions were found.")
        return

    # --- Filter Data (Now applied to the corrected set) ---
    print("\nApplying post-correction filters...")
    # The original filtering is now implicitly handled, but we can keep turn/player filtering.
    if feature_set == 'simplified' or feature_set == 'minimal_active_species':
         print(f"Filtering for player_to_move == 'p1' (for {feature_set} set)...")
         original_rows = len(X)
         X = X[X['player_to_move'] == 'p1'].copy()
         y_labels = y_labels[X.index] # Keep y aligned with X
         print(f"Rows after player filter: {len(X)} (Removed {original_rows - len(X)})")
         if X.empty: print(f"Error: No data found for player p1's switches."); return
    
    if min_turn > 0:
        original_rows = len(X)
        X = X[X['turn_number'] >= min_turn].copy()
        y_labels = y_labels[X.index] # Keep y aligned with X
        print(f"Rows after turn filter: {len(X)} (Removed {original_rows - len(X)})")
        if X.empty: print("Error: No data remaining after turn filtering."); return

    # --- Create MULTI-CLASS Target Variable ---
    print("\nCreating multi-class target variable 'switch_target'...")
    try:
        # We already have our labels in y_labels, now we just process them.
        y_labels = y_labels.str.split(':', n=1).str[1]
        print(f"Found {y_labels.nunique()} unique Pokémon switch targets.")
        if y_labels.nunique() < 2:
             print("Error: Only one type of Pokémon was switched into. Cannot train multi-class classifier.")
             return

        _, _, top_100_names = load_smogon_moves(USAGE_STATS_JSON)  
        # Standardize y_labels for matching (lowercase, no spaces/hyphens)
        y_labels_std = y_labels.str.lower().str.replace(' ', '').str.replace('-', '')

        # Filter to top 100 or map others to 'other'
        mask = y_labels_std.isin(top_100_names)
        y_labels_filtered = y_labels[mask].copy()
        X= X.loc[mask].copy()

        X['p1_active_slot'] = X.apply(lambda row: next((i for i in range(1,7) if row.get(f'p1_slot{i}_is_active', 0)==1), None), axis=1)
        # Create y_slot (drop rows where slot not found)
        y_slot = X.apply(lambda row: get_switch_slot(row, y_labels[row.name]), axis=1)
        valid_mask = y_slot.notnull()
        X = X[valid_mask]
        y_slot = y_slot[valid_mask].astype(int)  # 5 classes (bench slots)

        # *** Use LabelEncoder to convert species names to integer labels ***
        label_encoder = LabelEncoder()
        # y = pd.Series(label_encoder.fit_transform(y_labels_filtered), index=y_labels_filtered.index)
        y = label_encoder.fit_transform(y_slot)
        num_classes = len(label_encoder.classes_)
        print(f"Target variable created with {num_classes} classes.")

        y = pd.Series(y)

        # The y series has the same index as X, but we should reset it before filtering rare classes
        # This makes the rare class removal logic simpler
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        print(y.value_counts())

        print("\nChecking for rare classes before splitting...")
        min_samples_per_class = 3  # Set the minimum required samples for a class to be included

        class_counts = y.value_counts()
        rare_classes = class_counts[class_counts < min_samples_per_class].index
        
        if len(rare_classes) > 0:
            print(f"Warning: The following {len(rare_classes)} classes have fewer than 3 samples and will be removed:\n  {list(rare_classes)}")
            mask = ~y.isin(rare_classes)
            X = X.loc[mask].reset_index(drop=True)
            y = y.loc[mask].reset_index(drop=True)
            print(f"Removed {mask.size - mask.sum()} rows corresponding to rare classes. New total rows: {len(y)}")
        else:
            print("No rare classes found. All classes have sufficient samples for splitting.")

        # Save the label encoder
        model_suffix = f"{feature_set}_moves" if feature_set == 'simplified' else feature_set
        encoder_path = f'switch_target_predictor_label_encoder_{model_suffix}.joblib'
        joblib.dump(label_encoder, encoder_path)
        print(f"Label encoder saved to {encoder_path}")

    except Exception as e:
        print(f"An unexpected error occurred during target variable creation: {e}")
        import traceback
        traceback.print_exc()
        returnxc()

    # --- Feature Selection  ---
    numerical_features = []
    categorical_features = []
    if feature_set == 'minimal_active_species':
            print("\n--- Using MINIMAL ACTIVE SPECIES feature set ---")
            active_species_data = []
            # CRITICAL CHANGE: Iterate over the filtered 'X', not the original 'df'.
            for idx, row in X.iterrows():
                p1_active_species, p2_active_species = 'Unknown', 'Unknown'
                for i_slot in range(1, 7):
                    if row.get(f'p1_slot{i_slot}_is_active', 0) == 1: p1_active_species = row.get(f'p1_slot{i_slot}_species', 'Unknown')
                for i_slot in range(1, 7):
                    if row.get(f'p2_slot{i_slot}_is_active', 0) == 1: p2_active_species = row.get(f'p2_slot{i_slot}_species', 'Unknown')
                # Use the index from the iterated DataFrame 'X'
                active_species_data.append({'original_index': idx, 'p1_active_species': p1_active_species, 'p2_active_species': p2_active_species})
            
            X_minimal = pd.DataFrame(active_species_data).set_index('original_index')
            X = X_minimal # Overwrite X with the newly constructed feature set
            categorical_features = ['p1_active_species', 'p2_active_species']

    elif feature_set == 'simplified':
        print("\n--- Using REDEFINED SIMPLIFIED feature set + REVEALED MOVES ---")
        selected_columns = ['turn_number', 'last_move_p1', 'last_move_p2', 'field_weather']
        for i in range(1, 7):
            for player in ['p1', 'p2']:
                selected_columns.extend([f'{player}_slot{i}_species', f'{player}_slot{i}_hp_perc',f'{player}_slot{i}_status', f'{player}_slot{i}_is_active',f'{player}_slot{i}_is_fainted'])
        hazard_types = ['stealthrock', 'spikes', 'toxicspikes', 'stickyweb']
        for player in ['p1', 'p2']:
             for hazard in hazard_types: selected_columns.append(f'{player}_hazard_{hazard}')
        side_cond_types = ['reflect', 'lightscreen', 'auroraveil', 'tailwind']
        for player in ['p1', 'p2']:
            for cond in side_cond_types: selected_columns.append(f'{player}_side_{cond}')
        
        valid_selected_columns = [col for col in selected_columns if col in X.columns]
        # CRITICAL CHANGE: Select from the filtered 'X', not 'df'.
        X_simplified_base = X[valid_selected_columns].copy()

        active_data_list = []
        base_active_features = ['species', 'hp_perc', 'status', 'terastallized']
        # CRITICAL CHANGE: Iterate over the filtered 'X', not 'df'.
        for idx, row in X.iterrows():
            active_p1_slot, active_p2_slot = -1, -1
            for i_slot in range(1, 7):
                if row.get(f'p1_slot{i_slot}_is_active', 0) == 1: active_p1_slot = i_slot
                if row.get(f'p2_slot{i_slot}_is_active', 0) == 1: active_p2_slot = i_slot
            row_active_data = {'original_index': idx}
            for player, active_slot in [('p1', active_p1_slot), ('p2', active_p2_slot)]:
                for feat in base_active_features: row_active_data[f'{player}_active_{feat}'] = row.get(f'{player}_slot{active_slot}_{feat}', None) if active_slot != -1 else None
            active_data_list.append(row_active_data)
        
        active_df = pd.DataFrame(active_data_list).set_index('original_index')
        
        revealed_move_cols = sorted([col for col in X.columns if col.endswith('_revealed_moves')])
        all_revealed_moves_binary_cols = []
        if revealed_move_cols:
            all_revealed_moves = set()
            for col in revealed_move_cols:
                # CRITICAL CHANGE: Use filtered 'X' to find unique moves
                unique_in_col = X[col].dropna().astype(str).str.split(',').explode().unique()
                all_revealed_moves.update(m.strip() for m in unique_in_col if m and m.strip() not in ['none', 'error_state', '', 'nan'])
            
            unique_moves_list = sorted(list(all_revealed_moves))
            def sanitize_name(name): return re.sub(r'[^a-zA-Z0-9]+', '_', name).lower()
            new_move_cols_data = {}
            for base_col in revealed_move_cols:
                # CRITICAL CHANGE: Use filtered 'X' to build move features
                revealed_sets = X.loc[X_simplified_base.index, base_col].fillna('none').astype(str).str.split(',').apply(set)
                for move in unique_moves_list:
                    new_col_name = f"{base_col}_{sanitize_name(move)}"
                    new_move_cols_data[new_col_name] = revealed_sets.apply(lambda move_set: 1 if move in move_set else 0).astype(np.int8)
                    all_revealed_moves_binary_cols.append(new_col_name)
            
            if new_move_cols_data:
                 move_features_df = pd.DataFrame(new_move_cols_data)
                 X = pd.concat([X_simplified_base, active_df, move_features_df], axis=1)
            else: X = pd.concat([X_simplified_base, active_df], axis=1)
        else: X = pd.concat([X_simplified_base, active_df], axis=1)

        numerical_features = [f for f in X.columns if ('hp_perc' in f or 'is_active' in f or 'is_fainted' in f or 'turn_number' in f or 'hazard' in f or 'side' in f or 'terastallized' in f)]
        categorical_features = [f for f in X.columns if ('species' in f or 'status' in f or 'last_move' in f or 'field_weather' in f)]
        if all_revealed_moves_binary_cols: numerical_features.extend(all_revealed_moves_binary_cols)
        numerical_features = sorted(list(set(f for f in numerical_features if f in X.columns)))
        categorical_features = sorted(list(set(f for f in categorical_features if f in X.columns and f not in numerical_features)))
    
    elif feature_set == 'full':
        print("\n--- Using FULL feature set ---")
        # CRITICAL CHANGE: 'X' is already what we want. We just need to drop columns
        # that aren't features and re-identify the feature types.
        X = X.drop(columns=['replay_id', 'action_taken', 'battle_winner', 'player_to_move'], errors='ignore')
        numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    else:
        print(f"Error: Invalid feature_set '{feature_set}'.")
        return

    # Now that X is correctly constructed from the filtered data, we can safely delete the original large dataframe
    del df; gc.collect()

    print(f"\nFinal feature counts for '{feature_set}' set: Numerical: {len(numerical_features)}, Categorical: {len(categorical_features)}")
    if X.empty: print("Error: Feature DataFrame X is empty."); return

    # --- Split Data (Stratification now happens on the multi-class target) ---
    print("\nSplitting data into Train, Validation, Test sets...")
    try:
        # Stratification is even more important for multi-class, especially with rare switch targets
        X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=test_split_size, random_state=42, stratify=y)
        val_size_relative = val_split_size / (1.0 - test_split_size)
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=val_size_relative, random_state=42, stratify=y_train_full)
        print("Successfully split data with stratification.")
    except Exception as e:
        print(f"Error during data splitting (check if any class has only 1 sample): {e}. Cannot proceed.")
        return

    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")
    del X, y, X_train_full, y_train_full; gc.collect()

    # --- Calculate Class Weights (works for multi-class out of the box) ---
    print("\nCalculating class weights for handling imbalance...")
    class_weights_values = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {int(cls): float(weight) for cls, weight in zip(np.unique(y_train), class_weights_values)}
    print(f"Class weights calculated for {len(class_weight_dict)} classes.")

    # --- Preprocessing  ---
    feature_lists_path = f'switch_target_predictor_feature_lists_{model_suffix}.joblib'
    preprocessor_path = f'switch_target_predictor_tf_preprocessor_{model_suffix}.joblib'
    final_train_cols = X_train.columns.tolist()
    numerical_features = [f for f in numerical_features if f in final_train_cols]
    categorical_features = [f for f in categorical_features if f in final_train_cols]
    joblib.dump({'numerical_features': numerical_features, 'categorical_features': categorical_features}, feature_lists_path)
    X_train_processed, X_val_processed, X_test_processed = None, None, None

    if model_type == 'tensorflow':
        print("\nSetting up TF preprocessing pipeline (OneHotEncoder + Scaler)...")
        transformers = []
        if numerical_features: transformers.append(('num', StandardScaler(), numerical_features))
        if categorical_features: transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_features))
        if not transformers: print("Error: No features to process!"); return
        preprocessor = ColumnTransformer(transformers=transformers, remainder='drop', sparse_threshold=0.3)
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        X_test_processed = preprocessor.transform(X_test)
        joblib.dump(preprocessor, preprocessor_path)
        del X_train, X_val, X_test; gc.collect()
    elif model_type == 'lightgbm':
        print("\nPreprocessing for LGBM will occur inside its training function.")
        X_train_processed, X_val_processed, X_test_processed = X_train, X_val, X_test
    else:
        print(f"Error: Unknown model_type '{model_type}'"); return

    # --- Train Selected Model ---
    print(f"\n--- Initiating {model_type.upper()} Model Training ({model_suffix} features, Multi-Class Switch Target) ---")
    if model_type == 'tensorflow':
        train_tensorflow_switch_target_predictor(X_train_processed, X_val_processed, X_test_processed,
                                               y_train, y_val, y_test,
                                               class_weight_dict,
                                               num_classes, # Pass num_classes
                                               epochs, batch_size, learning_rate,
                                               model_suffix=model_suffix)
    elif model_type == 'lightgbm':
        train_lgbm_switch_target_predictor(X_train_processed, X_val_processed, X_test_processed,
                                     y_train, y_val, y_test,
                                     numerical_features, categorical_features,
                                     class_weight_dict,
                                     num_classes, label_encoder, # Pass num_classes and the encoder
                                     model_suffix=model_suffix,
                                     perform_hpo=args.perform_hpo,
                                     n_hpo_trials=args.n_hpo_trials)
    print("\n--- Switch Target Predictor Training Script Finished ---")


# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a multi-class predictor to determine WHICH Pokémon a player will switch to.")
    parser.add_argument("parquet_file", type=str, help="Path to the input Parquet file.")
    parser.add_argument("--model_type", choices=['tensorflow', 'lightgbm'], default='lightgbm', help="Type of model to train.")
    parser.add_argument("--feature_set", choices=['full', 'simplified','minimal_active_species','simplified2'], default='simplified', help="Feature set to use.")
    parser.add_argument("--min_turn", type=int, default=1, help="Minimum turn number to include (default: 1).")
    parser.add_argument("--test_split", type=float, default=0.2, help="Fraction for test set (default: 0.2).")
    parser.add_argument("--val_split", type=float, default=0.15, help="Fraction for validation set (relative to train data, default: 0.15).")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs (TF only).")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size (TF only).")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (TF only).")
    parser.add_argument("--perform_hpo", action='store_true', help="Perform Optuna hyperparameter optimization for LightGBM.")
    parser.add_argument("--n_hpo_trials", type=int, default=50, help="Number of trials for Optuna HPO.")
    args = parser.parse_args()

    if not (0 < args.test_split < 1 and 0 < args.val_split < 1):
        print("Error: split sizes must be between 0 and 1."); exit(1)

    # Call the main training function
    run_switch_target_training(
        parquet_path=args.parquet_file,
        model_type=args.model_type,
        feature_set=args.feature_set,
        min_turn=args.min_turn,
        test_split_size=args.test_split,
        val_split_size=args.val_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )