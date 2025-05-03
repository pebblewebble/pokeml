import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb # Import LightGBM
import argparse
import os
import joblib # To save the preprocessor

# Suppress TensorFlow warnings (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# --- Function for TensorFlow Model ---
def train_tensorflow_predictor(X_train_processed, X_val_processed, X_test_processed, # Now receives PROCESSED X data
                               y_train, y_val, y_test, # Original y splits (pandas Series or NumPy)
                               class_weight_dict,
                               epochs=20, batch_size=128, learning_rate=0.001):
    """Trains and evaluates the TensorFlow MLP model. Converts y to NumPy array."""
    print("\n--- Training TensorFlow Model ---")

    # --- Convert y splits to NumPy arrays ---
    y_train_np = y_train.to_numpy()
    y_val_np = y_val.to_numpy()
    y_test_np = y_test.to_numpy()
    # ---------------------------------------

    # --- Build the TensorFlow Model (Uses shape from processed X) ---
    print("Building the TensorFlow model...")
    input_shape = (X_train_processed.shape[1],)
    if input_shape[0] == 0: print("Error: TF Processed data has 0 features."); return None, None

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'), tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.summary()

    # --- Compile the Model ---
    print(f"Compiling the TF model with learning rate: {learning_rate}")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # --- Train the Model (Use NumPy y data) ---
    print("Training the TF model with class weights...")
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train_processed,
        y_train_np, # <<< Use NumPy array
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val_processed, y_val_np), # <<< Use NumPy array
        callbacks=[early_stopping],
        class_weight=class_weight_dict,
        verbose=1
    )
    print("TF Training finished.")

    # --- Evaluate the Model (Use NumPy y data) ---
    print("\nEvaluating TF model on the test set...")
    loss, accuracy = model.evaluate(X_test_processed, y_test_np, verbose=0) # <<< Use NumPy array
    print(f"TF Test Loss: {loss:.4f}")
    print(f"TF Test Accuracy: {accuracy:.4f}")

    print("\nTF Classification Report:")
    y_pred_proba = model.predict(X_test_processed)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    print(classification_report(y_test_np, y_pred, target_names=['P2 Wins (0)', 'P1 Wins (1)'], zero_division=0)) # <<< Use NumPy array

    print("\nTF Confusion Matrix:")
    print(confusion_matrix(y_test_np, y_pred)) # <<< Use NumPy array

    print("Saving TensorFlow model...")
    model_save_path = 'pokemon_tf_win_predictor.keras' 
    model.save(model_save_path)
    print(f"TensorFlow model saved to {model_save_path}")

    return history, model

# --- Function for LightGBM Model ---
def train_lgbm_predictor(X_train, X_val, X_test, y_train, y_val, y_test,
                         numerical_features, categorical_features,
                         class_weight_dict):
    """Trains and evaluates a LightGBM model."""
    print("\n--- Training LightGBM Model ---")

    # --- Preprocessing (LGBM Specific: LabelEncode/astype(category)) ---
    # LGBM works best with integer-encoded categories or its specific Dataset object
    # For simplicity here, we'll use pandas category type directly if possible,
    # but label encoding might be needed if there are many categories causing issues.
    print("Preparing data for LightGBM (using category dtype)...")
    X_train_lgbm = X_train.copy()
    X_val_lgbm = X_val.copy()
    X_test_lgbm = X_test.copy()

    # Convert categorical columns to 'category' dtype for LGBM
    for col in categorical_features:
        X_train_lgbm[col] = X_train_lgbm[col].astype('category')
        # Ensure validation/test sets use the same categories learned from training set
        # This prevents errors if val/test have categories not seen in train
        X_val_lgbm[col] = pd.Categorical(X_val_lgbm[col], categories=X_train_lgbm[col].cat.categories)
        X_test_lgbm[col] = pd.Categorical(X_test_lgbm[col], categories=X_train_lgbm[col].cat.categories)

    # Optional: Scale numerical features (can sometimes help tree models too)
    if numerical_features:
        print("Scaling numerical features for LightGBM...")
        scaler = StandardScaler()
        X_train_lgbm[numerical_features] = scaler.fit_transform(X_train_lgbm[numerical_features])
        X_val_lgbm[numerical_features] = scaler.transform(X_val_lgbm[numerical_features])
        X_test_lgbm[numerical_features] = scaler.transform(X_test_lgbm[numerical_features])
        scaler_save_path = 'lgbm_scaler.joblib' # Use consistent name
        try:
            joblib.dump(scaler, scaler_save_path)
            print(f"LGBM scaler saved to {scaler_save_path}")
        except Exception as e:
            print(f"Error saving LGBM scaler: {e}")


    print(f"LGBM Processed shapes - Train: {X_train_lgbm.shape}, Val: {X_val_lgbm.shape}, Test: {X_test_lgbm.shape}")

    # --- Build and Train LightGBM Model ---
    print("Training LightGBM model with class weights...")
    lgbm_model = lgb.LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        class_weight=class_weight_dict, # Pass class weights
        random_state=42,
        n_jobs=-1 # Use all available CPU cores
        # Add other hyperparameters like n_estimators, learning_rate, num_leaves etc. for tuning
        # n_estimators=200,
        # learning_rate=0.05,
        # num_leaves=31
    )

    # Use early stopping with the validation set
    eval_set = [(X_val_lgbm, y_val)]
    callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=True)] # Stop if val_loss doesn't improve for 20 rounds

    lgbm_model.fit(
        X_train_lgbm,
        y_train,
        eval_set=eval_set,
        eval_metric='logloss', # or 'auc'
        callbacks=callbacks,
        # Explicitly list categorical features IF using category dtype doesn't work well
        # categorical_feature=[X_train_lgbm.columns.get_loc(col) for col in categorical_features]
        # OR use 'auto' if supported by your version: categorical_feature='auto'
    )
    print("LGBM Training finished.")

    # --- Evaluate LightGBM Model ---
    print("\nEvaluating LGBM model on the test set...")
    y_pred = lgbm_model.predict(X_test_lgbm)
    # y_pred_proba = lgbm_model.predict_proba(X_test_lgbm)[:, 1] # Prob for class 1

    accuracy = accuracy_score(y_test, y_pred)
    print(f"LGBM Test Accuracy: {accuracy:.4f}")

    print("\nLGBM Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['P2 Wins (0)', 'P1 Wins (1)'], zero_division=0))

    print("\nLGBM Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # --- Optional: Feature Importance ---
    try:
        print("\nLGBM Feature Importances (Top 20):")
        feature_importances = pd.Series(lgbm_model.feature_importances_, index=X_train_lgbm.columns)
        print(feature_importances.nlargest(20))
        # Plotting requires matplotlib
        # feature_importances.nlargest(20).plot(kind='barh', figsize=(10, 8))
        # plt.title("LightGBM Feature Importances")
        # plt.show()
    except Exception as e_imp:
        print(f"Could not display feature importances: {e_imp}")

    print("Saving LightGBM model...")
    lgbm_model_save_path = 'pokemon_lgbm_predictor.joblib'
    joblib.dump(lgbm_model, lgbm_model_save_path)
    print(f"LightGBM model saved to {lgbm_model_save_path}")

    return lgbm_model


# --- Main execution function ---
def run_training(parquet_path, model_type='tensorflow', min_turn=0, test_split_size=0.2, val_split_size=0.15, epochs=30, batch_size=128, learning_rate=0.0001):
    """Loads data, splits, preprocesses, and calls the appropriate model training function."""

    print(f"Loading data from: {parquet_path}")
    # ... (loading code) ...
    try:
        df = pd.read_parquet(parquet_path)
        print(f"Original data shape: {df.shape}")
    except Exception as e:
        print(f"Error loading Parquet file: {e}"); return

    print("\n--- Analyzing Last State Information ---")
    # Get the index of the row corresponding to the maximum turn number for each replay_id
    # Ensure turn_number is numeric if it wasn't already
    df['turn_number'] = pd.to_numeric(df['turn_number'], errors='coerce')
    last_state_indices = df.loc[df.groupby('replay_id')['turn_number'].idxmax()].index
    last_states_df = df.loc[last_state_indices].copy() # Create a copy to avoid SettingWithCopyWarning

    # Calculate fainted counts (adjust range based on max team size, usually 6)
    fainted_p1_cols = [f'p1_slot{i}_is_fainted' for i in range(1, 7) if f'p1_slot{i}_is_fainted' in last_states_df.columns]
    fainted_p2_cols = [f'p2_slot{i}_is_fainted' for i in range(1, 7) if f'p2_slot{i}_is_fainted' in last_states_df.columns]

    if fainted_p1_cols:
        last_states_df['p1_fainted_count'] = last_states_df[fainted_p1_cols].sum(axis=1)
        print("\nCrosstab: Battle Winner vs. P1 Fainted Count (Last State):")
        print(pd.crosstab(last_states_df['battle_winner'], last_states_df['p1_fainted_count']))

    if fainted_p2_cols:
        last_states_df['p2_fainted_count'] = last_states_df[fainted_p2_cols].sum(axis=1)
        print("\nCrosstab: Battle Winner vs. P2 Fainted Count (Last State):")
        print(pd.crosstab(last_states_df['battle_winner'], last_states_df['p2_fainted_count']))

    print("--------------------------------------\n")
    # --- Filter by Turn Number ---
    # ... (filtering code) ...
    if min_turn > 0:
        print(f"Filtering data for turns >= {min_turn}...")
        df = df[df['turn_number'] >= min_turn].copy()
        print(f"Shape after filtering turns: {df.shape}")
        if df.empty: print("Error: No data remaining after turn filtering."); return


    # --- Prepare Features and Target ---
    print("Preparing features and target...")
    df = df.dropna(subset=['battle_winner'])
    df = df[df['battle_winner'].isin(['p1', 'p2'])]
    if df.empty: print("Error: No valid 'p1'/'p2' winner rows remaining."); return

    # --- Explicitly set y dtype to standard int ---
    y = df['battle_winner'].apply(lambda x: 1 if x == 'p1' else 0).astype(int)
    # --------------------------------------------

    # feature_columns = [col for col in df.columns if col not in ['replay_id', 'turn_number', 'action_taken','battle_winner','_is_fainted']]
    base_exclude = ['replay_id', 'action_taken', 'battle_winner']
    fainted_cols_to_exclude = [col for col in df.columns if '_is_fainted' in col]
    species_cols_to_exclude = [col for col in df.columns if '_species' in col]
    revealed_cols_to_exclude = [col for col in df.columns if '_revealed_moves' in col]
    cols_to_exclude = base_exclude + fainted_cols_to_exclude + species_cols_to_exclude + revealed_cols_to_exclude

    feature_columns = [col for col in df.columns if col not in cols_to_exclude]
    X = df[feature_columns].copy()
    # X = df[feature_columns].copy()
    print(f"DEBUG: Columns included in features (X): {X.columns.tolist()}")

    features_save_path = 'feature_list.joblib'
    joblib.dump(feature_columns, features_save_path)
    print(f"Feature list saved to {features_save_path}")

    # ... (rest of feature prep: category conversion, NaN filling) ...
    if 'player_to_move' in X.columns: X['player_to_move'] = X['player_to_move'].fillna('unknown').astype('category')
    categorical_features = []
    object_cols = X.select_dtypes(include=['object']).columns
    for col in object_cols: X[col] = X[col].fillna('unknown'); X[col] = X[col].astype('category'); categorical_features.append(col)
    categorical_features.extend(X.select_dtypes(include=['category']).columns.tolist()); categorical_features = sorted(list(set(categorical_features)))
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    if numerical_features:
         nan_counts = X[numerical_features].isnull().sum(); cols_with_nan = nan_counts[nan_counts > 0].index.tolist()
         if cols_with_nan:
             print(f"Warning: Numerical columns have NaNs: {cols_with_nan}. Filling with median.")
             for col in cols_with_nan: X[col] = X[col].fillna(X[col].median())


    print(f"\nUsing {len(feature_columns)} features. Target shape: {y.shape}")
    # ... (print feature lists) ...

    # --- Split Data ---
    print("\nSplitting data...")
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=test_split_size, random_state=42, stratify=y)
    val_size_relative = val_split_size / (1 - test_split_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=val_size_relative, random_state=42, stratify=y_train_full)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print("\nVERIFY: Sample of Test Set Data:")
    print(X_test.head())
    print("\nVERIFY: Corresponding Test Set Labels:")
    print(y_test.head())

    # --- Calculate Class Weights (Ensure keys are standard int) ---
    print("\nCalculating class weights...")
    unique_classes = np.unique(y_train)
    class_weights_array = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train.values)
    class_weight_dict = {int(cls): weight for cls, weight in zip(unique_classes, class_weights_array)}
    # -----------------------------------------
    print(f"Class weights: {class_weight_dict}")


    # --- Preprocess X data based on model type ---
    X_train_processed, X_val_processed, X_test_processed = None, None, None
    preprocessor_path = None # Path to save preprocessor
    scaler_path_lgbm = None # Path for LGBM scaler

    if model_type == 'tensorflow':
        print("\nSetting up TF preprocessing pipeline (OneHotEncoder)...")
        numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
        transformers = []
        if numerical_features: transformers.append(('num', numerical_transformer, numerical_features))
        if categorical_features: transformers.append(('cat', categorical_transformer, categorical_features))
        if not transformers: print("Error: No features for TF preprocessor."); return
        print("VERIFY: Columns going into preprocessing:", X.columns.tolist())
        preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
        preprocessor_path = 'tf_preprocessor.joblib'

    elif model_type == 'lightgbm':
        # Preprocessing for LGBM is done inside its function, but we could save scaler here
        # Or do minimal preprocessing needed for both if desired (e.g., only scaling)
        print("\nPreprocessing for LGBM will occur inside its training function.")
        # We still need processed X placeholders if TF is not selected, or pass None
        # Let's just pass the original X splits to the LGBM function for simplicity
        scaler_path_lgbm = 'lgbm_scaler.joblib'
        pass # No pre-processing needed here for LGBM path

    else:
        print(f"Error: Unknown model_type '{model_type}'")
        return

    # Fit and transform X data for TF if selected
    if model_type == 'tensorflow':
        print("Applying TF preprocessing...")
        try:
            X_train_processed = preprocessor.fit_transform(X_train)
            print((X_train_processed.shape[1]))
            X_val_processed = preprocessor.transform(X_val)
            X_test_processed = preprocessor.transform(X_test)
            print(f"TF Processed shapes - Train: {X_train_processed.shape}, Val: {X_val_processed.shape}, Test: {X_test_processed.shape}")
            joblib.dump(preprocessor, preprocessor_path)
            print(f"TF preprocessor saved to {preprocessor_path}")
        except Exception as e:
            print(f"Error during TF preprocessing: {e}"); return


    # --- Train Selected Model ---
    if model_type == 'tensorflow':
        if preprocessor_path is None or not os.path.exists(preprocessor_path):
             print(f"Error: TF Preprocessor file '{preprocessor_path}' not found or not saved correctly.")
             return
         # Pass PROCESSED X data but ORIGINAL y splits
        train_tensorflow_predictor(X_train_processed, X_val_processed, X_test_processed,
                                   y_train, y_val, y_test, # Pass original y Series
                                   class_weight_dict,
                                   epochs, batch_size, learning_rate)   
        joblib.dump(feature_columns, 'feature_list.joblib')

    elif model_type == 'lightgbm':
        if numerical_features and (scaler_path_lgbm is None or not os.path.exists(scaler_path_lgbm)):
             print(f"Warning: LGBM Scaler file '{scaler_path_lgbm}' not found or not saved.")
         # Pass ORIGINAL X and y splits
        train_lgbm_predictor(X_train, X_val, X_test, y_train, y_val, y_test,
                             numerical_features, categorical_features, class_weight_dict)
        joblib.dump(feature_columns, 'feature_list.joblib')


# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model (TF or LGBM) to predict Pokemon battle winner.")
    parser.add_argument("parquet_file", type=str, help="Path to the input Parquet file.")
    parser.add_argument("--model_type", choices=['tensorflow', 'lightgbm'], default='tensorflow', help="Type of model to train.")
    parser.add_argument("--min_turn", type=int, default=0, help="Minimum turn number to include in training data (0 for all turns).")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (TensorFlow only).")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training (TensorFlow only).")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for Adam optimizer (TensorFlow only).")

    args = parser.parse_args()

    run_training(
        parquet_path=args.parquet_file,
        model_type=args.model_type,
        min_turn=args.min_turn,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )