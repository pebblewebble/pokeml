import asyncio
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import argparse
import os
import gc
from collections import Counter
import warnings
import json
import re # Added for parsing/mapping helpers
import copy # Added for state copying if needed

# --- poke-env ---
try:
    import poke_env
    from poke_env.player import Player, Gen9EnvSinglePlayer
    from poke_env import AccountConfiguration
    # --- ADD/MODIFY THESE SPECIFIC IMPORTS ---
    from poke_env.environment.move import Move
    from poke_env.environment.pokemon import Pokemon
    from poke_env.player import DefaultBattleOrder
    from poke_env.player import BattleOrder
    # You might need others depending on your full logic, but these cover the error
    # from poke_env.environment.side_condition import SideCondition
    # from poke_env.environment.field import Field
    # --- END SPECIFIC IMPORTS ---
    from poke_env import ShowdownServerConfiguration
except ImportError:
    print("Error: poke-env library not found. Please install it: pip install poke-env")
    exit(1)


# Suppress PerformanceWarnings temporarily if desired
# warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning) # Ignore pandas 3.0 warnings


# ===========================================
# Configuration
# ===========================================
MODEL_PATH = "action_lgbm_model_v4_medium_move_only.txt"
INFO_PATH = "action_lgbm_feature_info_v4_medium_move_only.joblib"
SCALER_PATH = "action_lgbm_scaler_v4_medium_move_only.joblib" # Optional if no scaler used/needed by model
ENCODER_PATH = "action_label_encoder_v4_medium_move_only.joblib"
USAGE_STATS_JSON = "gen9ou-0.json" # Path to your Smogon usage stats

# Showdown Credentials (Replace or use environment variables!)
SHOWDOWN_USERNAME = os.environ.get("SHOWDOWN_USER", "www31")
SHOWDOWN_PASSWORD = os.environ.get("SHOWDOWN_PASS", "vimvimvim333")

BATTLE_FORMAT = "gen9ou"
LOG_LEVEL = 20 # 10=DEBUG, 20=INFO, 30=WARNING

team="""
Great Tusk @ Heavy-Duty Boots
Ability: Protosynthesis
Tera Type: Water
EVs: 252 HP / 4 Atk / 252 Def
Impish Nature
- Stealth Rock
- Rapid Spin
- Headlong Rush
- Knock Off

Kingambit @ Black Glasses
Ability: Supreme Overlord
Tera Type: Dark
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Kowtow Cleave
- Sucker Punch
- Iron Head
- Swords Dance

Gholdengo @ Choice Scarf
Ability: Good as Gold
Tera Type: Steel
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Make It Rain
- Shadow Ball
- Trick
- Focus Blast

Gliscor @ Toxic Orb
Ability: Poison Heal
Tera Type: Water
EVs: 244 HP / 144 Def / 120 SpD
Impish Nature
- Spikes
- Earthquake
- Knock Off
- Protect

Heatran @ Leftovers
Ability: Flash Fire
Tera Type: Grass
EVs: 252 HP / 188 SpD / 68 Spe
Calm Nature
- Magma Storm
- Earth Power
- Taunt
- Protect

Dragapult @ Heavy-Duty Boots
Ability: Infiltrator
Tera Type: Ghost
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Shadow Ball
- Draco Meteor
- U-turn
- Thunderbolt
"""

# ===========================================
# Utility Functions (Prediction & Helpers)
# ===========================================

# --- Artifact Loading ---
def load_artifacts(model_path, info_path, scaler_path, encoder_path):
    """Loads prediction model artifacts."""
    lgbm_model = None
    feature_info = None
    scaler = None
    label_encoder = None
    print("-" * 20)
    print("Loading artifacts...")
    try:
        print(f"  Model: {model_path}")
        if not os.path.exists(model_path): raise FileNotFoundError(f"Model file not found: {model_path}")
        lgbm_model = lgb.Booster(model_file=model_path)

        print(f"  Feature Info: {info_path}")
        if not os.path.exists(info_path): raise FileNotFoundError(f"Feature info file not found: {info_path}")
        feature_info = joblib.load(info_path)

        print(f"  Scaler: {scaler_path}")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print("    Scaler loaded.")
        else:
            print("    Scaler not found or not applicable.")

        print(f"  Label Encoder: {encoder_path}")
        if not os.path.exists(encoder_path): raise FileNotFoundError(f"Label encoder file not found: {encoder_path}")
        label_encoder = joblib.load(encoder_path)

        print("Artifacts loaded successfully.")
        print("-" * 20)
        return lgbm_model, feature_info, scaler, label_encoder
    except FileNotFoundError as e: print(f"Error loading artifacts: {e}. Check paths."); exit(1)
    except Exception as e: print(f"Unexpected error loading artifacts: {e}"); exit(1)

# --- Smogon Moves Loading ---
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
        return pokemon_valid_moves, metagame

    except FileNotFoundError:
        print(f"  Error: Smogon JSON file not found at '{json_filepath}'")
        raise
    except json.JSONDecodeError as e:
        print(f"  Error: Failed to decode Smogon JSON file '{json_filepath}'. Invalid JSON: {e}")
        raise
    except Exception as e:
        print(f"  Error: An unexpected error occurred while loading Smogon JSON: {e}")
        raise

# --- Data Preparation (Adapted from predict_action.py) ---
# This function prepares the single row DataFrame for the model
# predict_action.py

def prepare_input_data_medium(df_input_raw, feature_info, scaler):
    """Prepares the input DataFrame row for the 'medium' feature set model."""
    print("Preparing input data row using 'medium' feature set logic...")
    if df_input_raw.empty:
        print("  Input DataFrame is empty, cannot prepare.")
        return None

    # --- Defensive check ---
    if len(df_input_raw) != 1:
        print(f"  ERROR in prepare_input_data_medium: Expected single row DataFrame, but got {len(df_input_raw)} rows.")
        return None
    # --- End defensive check ---

    row_index = df_input_raw.index[0]
    # Work with the input row as a Series for easier access
    input_row_series = df_input_raw.loc[row_index]

    all_model_features = feature_info.get('feature_names_in_order', [])
    if not all_model_features: print("Error: 'feature_names_in_order' missing."); return None
    numerical_trained = feature_info.get('numerical_features', [])
    categorical_trained = feature_info.get('categorical_features', [])
    category_map = feature_info.get('category_map', {})
    default_cat_fill = 'Unknown'

    # --- Multi-Hot Encode Active Revealed Moves (from String Column) ---
    print("  Processing active revealed moves string column...")
    active_revealed_move_cols_str = ['p1_active_revealed_moves_str', 'p2_active_revealed_moves_str']
    known_moves = set()
    prefix1, prefix2 = "p1_active_revealed_move_", "p2_active_revealed_move_"
    # Infer known moves from the trained feature names
    for col_name in all_model_features:
        move_name = None
        sanitized_prefix = None
        if col_name.startswith(prefix1):
            move_name = col_name[len(prefix1):]
            sanitized_prefix = prefix1
        elif col_name.startswith(prefix2):
            move_name = col_name[len(prefix2):]
            sanitized_prefix = prefix2
        if move_name and sanitized_prefix:
            # Basic desanitization (match training script if more complex)
            original_move_name = move_name.replace('_', ' ')
            known_moves.add(original_move_name)

    # Dictionary to hold the calculated binary move features for the row
    new_binary_move_data = {}
    if not known_moves:
        print("  Warning: Could not determine known moves from training features for one-hot encoding.")
    else:
        print(f"  Found {len(known_moves)} known move targets for one-hot encoding.")
        known_moves_list = sorted(list(known_moves))

        for base_col_str in active_revealed_move_cols_str:
            # Check if the source string column exists in the input row
            if base_col_str not in input_row_series.index:
                # print(f"    Debug: Source move string column '{base_col_str}' not found.")
                continue

            player_prefix = base_col_str.split('_')[0]
            new_col_prefix = f"{player_prefix}_active_revealed_move"

            moves_str = input_row_series.get(base_col_str, 'none') # Safely get value
            revealed_set = set(moves_str.split(',')) if moves_str and moves_str != 'none' else set()

            # Generate 0/1 values for known moves based on the revealed set
            for move in known_moves_list:
                # Sanitize exactly as in training
                sanitized_move_name = move.replace(' ', '_').replace('-', '_').replace(':', '').replace('%', 'perc')
                expected_col_name = f"{new_col_prefix}_{sanitized_move_name}"

                # Only store data for binary columns the model actually expects
                if expected_col_name in all_model_features:
                    new_binary_move_data[expected_col_name] = 1 if move in revealed_set else 0

        print(f"  Generated {len(new_binary_move_data)} new binary active move features data.")

    # --- Construct the Final Data Row Dictionary ---
    # This dictionary will hold the final values for all expected features
    final_row_data = {}

    print("  Constructing final feature row...")
    for col in all_model_features:
        if col in new_binary_move_data:
            # Use the newly calculated binary value
            final_row_data[col] = new_binary_move_data[col]
        elif col in input_row_series.index and col not in active_revealed_move_cols_str:
            # Use data from the original input row IF it exists AND it's not the temp string col
            final_row_data[col] = input_row_series[col]
        else:
            # Column is missing from input and wasn't generated as a binary feature
            # Set a default value based on naming conventions
            if any(p in col for p in ['_hp_perc', '_boost_', '_hazard_', '_side_', '_is_active', '_is_fainted', '_terastallized']) or col == 'turn_number' or col.startswith(('p1_active_revealed_move_', 'p2_active_revealed_move_')):
                final_row_data[col] = 0 # Default for numerical/binary
            elif any(p in col for p in ['_species', '_status', '_tera_type', 'field_', 'last_move']): # Note: removed *_revealed_moves string patterns
                final_row_data[col] = default_cat_fill # Default for string/categorical
            else:
                # Fallback default for any other unexpected missing column
                # print(f"    Debug: Setting default 0 for unexpected missing column '{col}'")
                final_row_data[col] = 0

    # --- Create Final DataFrame (Single Row) ---
    # Create directly from the dictionary, ensuring correct column order
    # Wrap values in lists to make it a single-row DataFrame
    try:
        final_row_data_for_df = {col: [final_row_data[col]] for col in all_model_features}
        X_final = pd.DataFrame(final_row_data_for_df, index=[row_index])
        print(f"  Constructed DataFrame with shape: {X_final.shape}")
        # --- DEBUG: Check for duplicates immediately after creation ---
        duplicate_cols_final = X_final.columns[X_final.columns.duplicated()].tolist()
        if duplicate_cols_final:
             print(f"  CRITICAL ERROR: Duplicates still present after final construction: {duplicate_cols_final}")
             return None # Stop if construction failed
        # --- END DEBUG ---
    except Exception as e_construct:
        print(f"  ERROR constructing final DataFrame: {e_construct}")
        return None


    # --- Preprocessing (NaNs, Dtypes, Scaling) on the FINAL DataFrame ---
    # This ensures preprocessing happens on the structure the model expects
    print("  Applying preprocessing (NaNs, Dtypes, Scaling) on final structure...")

    # Handle NaNs (less likely now, but important safeguard)
    # Iterating through X_final.columns ensures we only check existing columns
    for col in X_final.columns:
        value = X_final.loc[row_index, col] # Should be scalar now
        is_null = pd.isna(value)

        if is_null:
            # print(f"    Handling NaN found in column '{col}' after final construction.")
            if col in numerical_trained:
                fill_value = 0
                if col.endswith('_hp_perc'): fill_value = 100
                X_final.loc[row_index, col] = fill_value
            elif col in categorical_trained:
                X_final.loc[row_index, col] = default_cat_fill
            else: # Likely a binary column if NaN (unlikely) or other
                X_final.loc[row_index, col] = 0

    # Apply Categorical Dtypes
    print("  Applying categorical dtypes...")
    for col in categorical_trained:
         # Check if the column exists in the final DataFrame (it should)
         if col in X_final.columns:
              X_final[col] = X_final[col].astype(str) # Ensure string first
              if col in category_map:
                  # Use categories from training
                  known_categories = category_map[col].categories.tolist()
                  if default_cat_fill not in known_categories: known_categories.append(default_cat_fill)
                  cat_type = pd.CategoricalDtype(categories=known_categories, ordered=False)
                  try:
                       # Check if current value is valid before casting
                       current_value = X_final.loc[row_index, col]
                       if current_value not in cat_type.categories:
                            # print(f"    Warning: Value '{current_value}' in '{col}' not in known categories. Setting to '{default_cat_fill}'.")
                            X_final.loc[row_index, col] = default_cat_fill
                       # Now cast to the specific categorical type
                       X_final[col] = X_final[col].astype(cat_type)
                  except Exception as e:
                       print(f"      ERROR: Failed to cast '{col}' to known categorical type: {e}. Trying fallback.")
                       try: X_final[col] = X_final[col].astype('category') # Basic fallback cast
                       except Exception as e2: print(f"       Fallback cast failed for {col}: {e2}")
              else:
                   # No specific category map, just make it categorical
                   try: X_final[col] = X_final[col].astype('category')
                   except Exception as e: print(f"     Error converting {col} to basic category: {e}")

    # Scale Numerical Features
    # Get the list of numerical features that are actually present in the final DataFrame
    features_to_scale = [f for f in numerical_trained if f in X_final.columns]
    if scaler and features_to_scale:
        print(f"  Scaling {len(features_to_scale)} numerical features...")
        try:
            # Apply scaling transformation
            X_final[features_to_scale] = scaler.transform(X_final[features_to_scale])
        except ValueError as e:
            print(f"    ERROR during scaling: {e}. Check feature consistency.")
            print(f"    Scaler expected features: {scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else 'N/A'}")
            print(f"    Actual columns being scaled: {features_to_scale}")
            return None # Stop if scaling fails
        except Exception as e:
             print(f"    UNEXPECTED ERROR during scaling: {e}")
             return None
    elif scaler: print("  Scaler loaded, but no numerical features defined/found to scale.")
    else: print("  No scaler loaded or needed.")


    print(f"Final processed shape for prediction: {X_final.shape}")
    # print(f"Final dtypes:\n{X_final.dtypes}") # Uncomment for debug
    return X_final


# --- Prediction Function (Adapted from predict_action.py) ---
def predict_moves_with_filter(df_input, model, feature_info, scaler, label_encoder, pokemon_valid_moves, top_k=5):
    """
    Predicts moves using the model and filters results based on Smogon valid moves.
    Includes internal call to prepare_input_data_medium.
    """
    # --- Pre-check: Ensure player_to_move exists if needed by feature prep ---
    # The mapping function should add this.
    if 'player_to_move' not in df_input.columns:
        print("Warning: Input DataFrame is missing 'player_to_move'. Proceeding without it if possible.")
        # Ensure it doesn't break prepare_input_data_medium if that function relies on it

    # --- Prepare Data ---
    # Pass the single row DataFrame to the preparation function
    print("--- Online Input DataFrame Info ---")
    df_input.info()
    print("\n--- Online Input DataFrame Head ---")
    print(df_input.head())
    print("\n--- Online Input Data Types ---")
    print(df_input.dtypes)
    # Log unique values for a few key categorical columns
    if 'p1_active_species' in df_input.columns:
        print("\n--- Online Unique p1_active_species ---")
        print(df_input['p1_active_species'].unique())
    # Add similar logging for p2_active_species, statuses, tera_types etc.
    print("------------------------------------")
    X_processed = prepare_input_data_medium(df_input, feature_info, scaler)
    if X_processed is None:
        print("Error during data preparation.")
        return None # Return None to indicate failure

    # --- Make Predictions ---
    print("\nMaking predictions...")
    predictions_list = []
    try:
        # Identify categorical features *expected by the model* that are present
        categorical_features_for_lgbm = [
            f for f in feature_info.get('categorical_features', [])
            if f in X_processed.columns and pd.api.types.is_categorical_dtype(X_processed[f]) # Double check dtype
        ]
        if len(categorical_features_for_lgbm) != len(feature_info.get('categorical_features', [])):
             print("Warning: Mismatch between expected categorical features and prepared features.")
             print(f"  Expected: {feature_info.get('categorical_features', [])}")
             print(f"  Actual sent to LGBM: {categorical_features_for_lgbm}")

        print(f"  Passing {len(categorical_features_for_lgbm)} features as categorical to LGBM.")
        pred_probabilities = model.predict(X_processed, categorical_feature=categorical_features_for_lgbm)

        # --- Decode and Filter ---
        print("Decoding and filtering predictions based on Smogon data...")
        all_possible_actions = label_encoder.classes_
        non_move_actions = {action for action in all_possible_actions if not action.startswith('move:')}

        # Process the single prediction row
        if pred_probabilities.ndim == 2 and pred_probabilities.shape[0] == 1:
            row_probs = pred_probabilities[0] # Get the probabilities for the single input row
            num_classes = len(all_possible_actions)

            # --- Identify active Pokemon for filtering ---
            input_row = df_input.iloc[0] # Get the original input row for finding active species
            player_moving = input_row.get('player_to_move', 'p1') # Default to p1 if missing
            active_pokemon_species_mapped = 'Unknown'
            # Find active species from the mapped DataFrame features
            for i in range(1, 7):
                 active_col = f"{player_moving}_slot{i}_is_active"
                 species_col = f"{player_moving}_slot{i}_species"
                 if active_col in input_row and species_col in input_row and input_row[active_col] == 1:
                      active_pokemon_species_mapped = input_row[species_col]
                      break # Found the active one

            # Standardize the species name for Smogon lookup (lowercase, no spaces/hyphens)
            active_pokemon_lookup_key = active_pokemon_species_mapped.lower().replace(' ', '').replace('-','')

            # ++++++++++++++++++++++++++++++++++++++++++++
            print(f"  DEBUG: Predicting/Filtering for active Pokemon: '{active_pokemon_species_mapped}' (Lookup Key: '{active_pokemon_lookup_key}')")
            # ++++++++++++++++++++++++++++++++++++++++++++

            # --- Determine valid actions ---
            valid_actions_for_pokemon = set()
            if not pokemon_valid_moves:
                print(f"Warning: Smogon valid moves data is empty. Using unfiltered predictions.")
                valid_actions_for_pokemon = set(all_possible_actions) # Allow everything
            elif active_pokemon_species_mapped == 'Unknown' or active_pokemon_species_mapped == 'Absent':
                print(f"Warning: Active Pokemon species is '{active_pokemon_species_mapped}'. Using unfiltered predictions.")
                valid_actions_for_pokemon = set(all_possible_actions) # Allow everything
            elif active_pokemon_lookup_key not in pokemon_valid_moves:
                 # Check base form if lookup key failed (e.g., 'urshifu-rapidstrike' vs 'urshifu')
                 base_form_key = active_pokemon_lookup_key.split('-')[0]
                 if base_form_key in pokemon_valid_moves:
                      print(f"Note: Using base form '{base_form_key}' moves for '{active_pokemon_species_mapped}'.")
                      valid_moves_from_smogon = pokemon_valid_moves[base_form_key]
                      valid_actions_for_pokemon = valid_moves_from_smogon.union(non_move_actions)
                 else:
                      print(f"Warning: Active species '{active_pokemon_species_mapped}' (key: {active_pokemon_lookup_key}/{base_form_key}) not found in Smogon data. Allowing only non-moves.")
                      valid_actions_for_pokemon = non_move_actions # Allow only switches/etc.
            else:
                # Get moves known for this species + all non-move actions
                valid_moves_from_smogon = pokemon_valid_moves[active_pokemon_lookup_key]
                valid_actions_for_pokemon = valid_moves_from_smogon.union(non_move_actions)

            # --- Filter and Rank ---
            valid_action_probs = []
            for action_idx in range(num_classes):
                action_str = all_possible_actions[action_idx]
                if action_str in valid_actions_for_pokemon:
                    valid_action_probs.append((row_probs[action_idx], action_str))

            valid_action_probs.sort(key=lambda x: x[0], reverse=True)
            top_k_valid = valid_action_probs[:top_k]

            # Format output DataFrame for this single prediction
            result_dict = {
                'rank': list(range(1, len(top_k_valid) + 1)),
                # --- CHANGE THIS LINE ---
                # Store the full action string from item[1]
                'action_string': [item[1] for item in top_k_valid],
                # --- END CHANGE ---
                'probability': [item[0] for item in top_k_valid]
            }
            # The rest of the function stays the same (append df, return list)
            predictions_list.append(pd.DataFrame(result_dict))
            print("Decoding and filtering predictions complete.")
        else:
            print(f"Error: Unexpected prediction probability shape: {pred_probabilities.shape}")

    except Exception as e:
         print(f"Error during model prediction or filtering: {e}")
         import traceback
         traceback.print_exc()
         return None # Indicate error

    return predictions_list # Return list (should contain one DataFrame)


# ===========================================
# poke-env Player Class
# ===========================================

class PredictionPlayer(Player):
    # last_used_p1 = 'none'
    # last_used_p2 = 'none'
    latest_battle_message = ''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # --- Load Artifacts ONCE during player initialization ---
        # Make load_artifacts robust to potential errors
        try:
            self.model, self.feature_info, self.scaler, self.label_encoder = load_artifacts(
                MODEL_PATH, INFO_PATH, SCALER_PATH, ENCODER_PATH
            )
        except Exception as e:
             print(f"FATAL ERROR during artifact loading: {e}")
             # Decide how to handle - maybe raise exception to stop bot?
             raise RuntimeError("Failed to load prediction artifacts.") from e

        # --- Load Smogon Moves ---
        try:
            self.pokemon_valid_moves, _ = load_smogon_moves(USAGE_STATS_JSON)
            if not self.pokemon_valid_moves:
                 print("Warning: Smogon valid moves data is empty. Filtering will be skipped.")
        except Exception as e:
             print(f"ERROR loading Smogon moves: {e}. Filtering will be skipped.")
             self.pokemon_valid_moves = {} # Ensure it's an empty dict

        # --- Store Expected Features ---
        if not self.feature_info or 'feature_names_in_order' not in self.feature_info:
             raise RuntimeError("Feature info is missing or doesn't contain 'feature_names_in_order'.")
        self.expected_features = self.feature_info.get('feature_names_in_order')
        print(f"Player initialized. Expecting {len(self.expected_features)} features.")

    def _handle_battle_message(self, split_messages):
        print("YOOOOOOOOO")
        print(split_messages)
        return super()._handle_battle_message(split_messages)


    
    
    def map_battle_to_dataframe_row(self, battle: Gen9EnvSinglePlayer.battles) -> dict:
        """
        *** CRITICAL IMPLEMENTATION (REVISED) ***
        Converts the poke-env Battle object into a flat dictionary row suitable
        for the prediction model. Keys and formatting MUST match the output
        of process_replays.py.
        """
        HAZARD_CONDITIONS_MAP = {
            'stealthrock': 'stealth_rock', 'spikes': 'spikes',
            'toxicspikes': 'toxic_spikes', 'stickyweb': 'sticky_web'
        }
        SIDE_CONDITIONS_MAP = {
            'reflect': 'reflect', 'lightscreen': 'light_screen',
            'auroraveil': 'aurora_veil', 'tailwind': 'tailwind',
            'safeguard': 'safeguard', #'mist': 'mist', # Add if trained on these
            # Add other side conditions your parser extracts if needed
        }
        ALL_HAZARD_SUFFIXES = set(HAZARD_CONDITIONS_MAP.values())
        ALL_SIDE_SUFFIXES = set(SIDE_CONDITIONS_MAP.values())
        ALL_EXPECTED_HAZARD_KEYS = {f"{p}_hazard_{s}" for p in ['p1', 'p2'] for s in ALL_HAZARD_SUFFIXES}
        ALL_EXPECTED_SIDE_KEYS = {f"{p}_side_{s}" for p in ['p1', 'p2'] for s in ALL_SIDE_SUFFIXES}
        print(f"Mapping Battle state for turn {battle.turn}...")
        flat_state = {}

        # --- Basic Info (Consistent with Parser) ---
        flat_state['replay_id'] = battle.battle_tag
        flat_state['turn_number'] = battle.turn
        flat_state['player_to_move'] = 'p1' # Bot's perspective (Consistent)

        # --- Last Moves (Placeholder - Still requires separate tracking) ---
        flat_state['last_move_p1'] = 'none' # Placeholder - NEEDS EXTERNAL TRACKING
        flat_state['last_move_p2'] = 'none' # Placeholder - NEEDS EXTERNAL TRACKING

        # --- Player (p1 - Bot) Team State ---
        team = battle.team
        active = battle.active_pokemon
        for i in range(6): # Always generate 6 slots like parser
            slot_id = f"slot{i+1}"
            prefix = f"p1_{slot_id}"
            pkmn: Gen9EnvSinglePlayer.Pokemon = None
            team_list = list(team.values())
            if i < len(team_list):
                pkmn = team_list[i]

            if pkmn:
                # *** Normalization to match process_replays.py ***
                species_name = pkmn.species.title().lower().replace(" ","") if pkmn.species else 'Unknown'
                flat_state[f'{prefix}_species'] = species_name

                # HP/Status/Fainted: Consistent with parser logic
                flat_state[f'{prefix}_hp_perc'] = round(pkmn.current_hp_fraction * 100)
                status_str = pkmn.status.name.lower() if pkmn.status else 'none' # Parser also outputs lowercase status
                flat_state[f'{prefix}_status'] = status_str
                flat_state[f'{prefix}_is_active'] = int(pkmn == active)
                flat_state[f'{prefix}_is_fainted'] = int(pkmn.fainted)

                # Terastallization: Normalize tera type case
                is_tera = getattr(pkmn, 'is_terastallized', pkmn.is_dynamaxed) # is_dynamaxed likely placeholder
                flat_state[f'{prefix}_terastallized'] = int(is_tera)
                tera_type = getattr(pkmn, 'tera_type', None)
                # Use .title() for tera type name to match parser output ('Water' not 'water')
                tera_type_name = tera_type.name.title() if tera_type else 'none'
                flat_state[f'{prefix}_tera_type'] = tera_type_name

                # Boosts: Consistent with parser
                boosts = pkmn.boosts
                for stat in ['atk', 'def', 'spa', 'spd', 'spe']:
                    flat_state[f'{prefix}_boost_{stat}'] = boosts.get(stat, 0)

                # Revealed Moves: Normalize move name case to Title Case and format
                # Filter 'conversion' like parser might implicitly do, adjust if needed
                moves_set = {m.id.title() for m in pkmn.moves.values() if m.id != 'conversion'}
                flat_state[f'{prefix}_revealed_moves'] = ",".join(sorted(list(moves_set))) if moves_set else 'none'

            else:
                # Fill with 'Absent' defaults (Consistent with parser's handling of smaller teams)
                flat_state[f'{prefix}_species'] = 'Absent'
                flat_state[f'{prefix}_hp_perc'] = 0; flat_state[f'{prefix}_status'] = 'none'; flat_state[f'{prefix}_is_active'] = 0; flat_state[f'{prefix}_is_fainted'] = 1
                flat_state[f'{prefix}_terastallized'] = 0; flat_state[f'{prefix}_tera_type'] = 'none'
                for stat in ['atk', 'def', 'spa', 'spd', 'spe']: flat_state[f'{prefix}_boost_{stat}'] = 0
                flat_state[f'{prefix}_revealed_moves'] = 'none'

        # --- Opponent (p2) Team State ---
        opp_team = battle.opponent_team
        opp_active = battle.opponent_active_pokemon
        for i in range(6): # Always generate 6 slots
            slot_id = f"slot{i+1}"
            prefix = f"p2_{slot_id}"
            pkmn: Gen9EnvSinglePlayer.Pokemon = None
            opp_team_list = list(opp_team.values())
            if i < len(opp_team_list):
                pkmn = opp_team_list[i]

            if pkmn:
                # *** Normalization to match process_replays.py ***
                species_name = pkmn.species.title() if pkmn.species else 'Unknown'
                flat_state[f'{prefix}_species'] = species_name

                flat_state[f'{prefix}_hp_perc'] = round(pkmn.current_hp_fraction * 100)
                status_str = pkmn.status.name.lower() if pkmn.status else 'none'
                flat_state[f'{prefix}_status'] = status_str
                flat_state[f'{prefix}_is_active'] = int(pkmn == opp_active)
                flat_state[f'{prefix}_is_fainted'] = int(pkmn.fainted)

                is_tera = getattr(pkmn, 'is_terastallized', pkmn.is_dynamaxed)
                flat_state[f'{prefix}_terastallized'] = int(is_tera)
                tera_type = getattr(pkmn, 'tera_type', None)
                tera_type_name = tera_type.name.title() if tera_type else 'none'
                flat_state[f'{prefix}_tera_type'] = tera_type_name

                boosts = pkmn.boosts
                for stat in ['atk', 'def', 'spa', 'spd', 'spe']:
                    flat_state[f'{prefix}_boost_{stat}'] = boosts.get(stat, 0)

                # Revealed Moves: Normalize case and format
                moves_set = {m.id.title() for m in pkmn.moves.values() if m.id != 'conversion'} # Only known moves
                flat_state[f'{prefix}_revealed_moves'] = ",".join(sorted(list(moves_set))) if moves_set else 'none'

            else:
                # Fill with 'Absent' defaults
                flat_state[f'{prefix}_species'] = 'Absent'
                flat_state[f'{prefix}_hp_perc'] = 0; flat_state[f'{prefix}_status'] = 'none'; flat_state[f'{prefix}_is_active'] = 0; flat_state[f'{prefix}_is_fainted'] = 1
                flat_state[f'{prefix}_terastallized'] = 0; flat_state[f'{prefix}_tera_type'] = 'none'
                for stat in ['atk', 'def', 'spa', 'spd', 'spe']: flat_state[f'{prefix}_boost_{stat}'] = 0
                flat_state[f'{prefix}_revealed_moves'] = 'none'


        # --- Active Pokemon Details (for prepare_input_data_medium) ---
        # Copy details for P1 active Pokemon (using normalized values)
        active_p1_prefix = None
        if active:
            for i in range(len(list(team.values()))): # Find the index/slot of active
                if list(team.values())[i] == active:
                    active_p1_prefix = f"p1_slot{i+1}"
                    break
        if active_p1_prefix and f'{active_p1_prefix}_species' in flat_state:
            # Copy relevant fields, ensuring normalized case is used
            flat_state['p1_active_species'] = flat_state[f'{active_p1_prefix}_species'] # Already Title Case
            flat_state['p1_active_hp_perc'] = flat_state[f'{active_p1_prefix}_hp_perc']
            flat_state['p1_active_status'] = flat_state[f'{active_p1_prefix}_status'] # Already lowercase
            flat_state['p1_active_boost_atk'] = flat_state[f'{active_p1_prefix}_boost_atk']
            flat_state['p1_active_boost_def'] = flat_state[f'{active_p1_prefix}_boost_def']
            flat_state['p1_active_boost_spa'] = flat_state[f'{active_p1_prefix}_boost_spa']
            flat_state['p1_active_boost_spd'] = flat_state[f'{active_p1_prefix}_boost_spd']
            flat_state['p1_active_boost_spe'] = flat_state[f'{active_p1_prefix}_boost_spe']
            flat_state['p1_active_terastallized'] = flat_state[f'{active_p1_prefix}_terastallized']
            flat_state['p1_active_tera_type'] = flat_state[f'{active_p1_prefix}_tera_type'] # Already Title Case
            flat_state['p1_active_revealed_moves_str'] = flat_state[f'{active_p1_prefix}_revealed_moves'] # Already Title Case moves, comma-sep
        else:
            print("Warning: Could not find P1 active Pokemon details during mapping.")
            # Set defaults consistent with prepare_input expectations
            flat_state['p1_active_species'] = 'Unknown'; flat_state['p1_active_hp_perc'] = 100; flat_state['p1_active_status'] = 'none'
            flat_state['p1_active_boost_atk'] = 0; flat_state['p1_active_boost_def'] = 0; flat_state['p1_active_boost_spa'] = 0; flat_state['p1_active_boost_spd'] = 0; flat_state['p1_active_boost_spe'] = 0
            flat_state['p1_active_terastallized'] = 0; flat_state['p1_active_tera_type'] = 'none'; flat_state['p1_active_revealed_moves_str'] = 'none'

        # Copy details for P2 active Pokemon (using normalized values)
        active_p2_prefix = None
        if opp_active:
            for i in range(len(list(opp_team.values()))): # Find the index/slot of opponent active
                if list(opp_team.values())[i] == opp_active:
                    active_p2_prefix = f"p2_slot{i+1}"
                    break
        if active_p2_prefix and f'{active_p2_prefix}_species' in flat_state:
            # Copy relevant fields, ensuring normalized case is used
            flat_state['p2_active_species'] = flat_state[f'{active_p2_prefix}_species'] # Already Title Case
            flat_state['p2_active_hp_perc'] = flat_state[f'{active_p2_prefix}_hp_perc']
            flat_state['p2_active_status'] = flat_state[f'{active_p2_prefix}_status'] # Already lowercase
            flat_state['p2_active_boost_atk'] = flat_state[f'{active_p2_prefix}_boost_atk']
            flat_state['p2_active_boost_def'] = flat_state[f'{active_p2_prefix}_boost_def']
            flat_state['p2_active_boost_spa'] = flat_state[f'{active_p2_prefix}_boost_spa']
            flat_state['p2_active_boost_spd'] = flat_state[f'{active_p2_prefix}_boost_spd']
            flat_state['p2_active_boost_spe'] = flat_state[f'{active_p2_prefix}_boost_spe']
            flat_state['p2_active_terastallized'] = flat_state[f'{active_p2_prefix}_terastallized']
            flat_state['p2_active_tera_type'] = flat_state[f'{active_p2_prefix}_tera_type'] # Already Title Case
            flat_state['p2_active_revealed_moves_str'] = flat_state[f'{active_p2_prefix}_revealed_moves'] # Already Title Case moves, comma-sep
        else:
            print("Warning: Could not find P2 active Pokemon details during mapping.")
            # Set defaults consistent with prepare_input expectations
            flat_state['p2_active_species'] = 'Unknown'; flat_state['p2_active_hp_perc'] = 100; flat_state['p2_active_status'] = 'none'
            flat_state['p2_active_boost_atk'] = 0; flat_state['p2_active_boost_def'] = 0; flat_state['p2_active_boost_spa'] = 0; flat_state['p2_active_boost_spd'] = 0; flat_state['p2_active_boost_spe'] = 0
            flat_state['p2_active_terastallized'] = 0; flat_state['p2_active_tera_type'] = 'none'; flat_state['p2_active_revealed_moves_str'] = 'none'


        # --- Field State ---
        # Weather: Normalize case
        flat_state['field_weather'] = battle.weather.name.title() if battle.weather else 'none'

        # Terrain: Not directly available in poke-env Battle object - Set default matching parser's 'none' state.
        # NEEDS MANUAL TRACKING if model relies heavily on this feature.
        flat_state['field_terrain'] = 'none' # <<< MODIFIED LINE

        # Pseudo Weather: Also not directly available - Set default.
        # NEEDS MANUAL TRACKING if model relies heavily on this feature.
        flat_state['field_pseudo_weather'] = 'none' # Set default, NEEDS EXTERNAL TRACKING if used by model

        # Hazards & Side Conditions: Map from poke-env's lowercase keys to parser's feature names
        # Initialize all expected keys to 0 first
        for key in ALL_EXPECTED_HAZARD_KEYS.union(ALL_EXPECTED_SIDE_KEYS):
            flat_state[key] = 0

        # Player 1 Side
        for cond, count in battle.side_conditions.items():
            if cond in HAZARD_CONDITIONS_MAP:
                key = f'p1_hazard_{HAZARD_CONDITIONS_MAP[cond]}'
                flat_state[key] = count # Typically layers (1, 2, or 3)
            elif cond in SIDE_CONDITIONS_MAP:
                key = f'p1_side_{SIDE_CONDITIONS_MAP[cond]}'
                flat_state[key] = 1 if count > 0 else 0 # Typically just active (1) or not (0)

        # Player 2 Side
        for cond, count in battle.opponent_side_conditions.items():
            if cond in HAZARD_CONDITIONS_MAP:
                key = f'p2_hazard_{HAZARD_CONDITIONS_MAP[cond]}'
                flat_state[key] = count
            elif cond in SIDE_CONDITIONS_MAP:
                key = f'p2_side_{SIDE_CONDITIONS_MAP[cond]}'
                flat_state[key] = 1 if count > 0 else 0

        # --- Return the flat dictionary ---
        # print(f"Sample mapped state keys: {list(flat_state.keys())[:10]}...") # Debug
        # print(f"Sample mapped state values (first 10): {list(flat_state.values())[:10]}...") # Debug
        return flat_state


    def choose_move(self, battle: Gen9EnvSinglePlayer.battles):
        """Called by poke-env to choose an action."""
        print(f"\n>>> Turn {battle.turn}: Choosing Action for {battle.battle_tag} <<<")
        print(f"Available Moves: {[m.id for m in battle.available_moves]}")
        print(f"Available Switches: {[p.species for p in battle.available_switches]}")
        print(f"Force Switch: {battle.force_switch}")

        # --- Optimization Check ---
        if battle.force_switch and len(battle.available_switches) == 1:
            the_only_choice = battle.available_switches[0] # This is a Pokemon object
            print(f"Optimization: Force switch and only one switch available ('{the_only_choice.species}'). Choosing it.")
            # !!! FIX: Wrap the Pokemon object in a BattleOrder before returning !!!
            return self.create_order(the_only_choice) # <--- USE create_order HERE
        if not battle.available_switches and len(battle.available_moves) == 1:
            action = battle.available_moves[0]
            print(f"Optimization: Only one move available ('{action.id}'). Choosing it.")
            return action
        if battle.force_switch and not battle.available_moves and len(battle.available_switches) == 1:
             action = battle.available_switches[0]
             print(f"Optimization: Force switch and only one switch available ('{action.species}'). Choosing it.")
             return action
        # --- End Optimization Check ---


        # 1. Map the current poke-env Battle object to the DataFrame format
        try:
            flat_state_dict = self.map_battle_to_dataframe_row(battle)
            input_df = pd.DataFrame([flat_state_dict], index=[0])
        except Exception as e:
            print(f"ERROR during state mapping: {e}")
            import traceback
            traceback.print_exc()
            print("Choosing default random action due to mapping error.")
            return self.choose_random_move(battle)

        # 2. Call prediction function
        predictions_list = None
        try:
            if not all([self.model, self.feature_info, self.label_encoder, self.pokemon_valid_moves is not None]):
                 raise ValueError("One or more prediction components (model, info, encoder, smogon_moves) are missing.")

            predictions_list = predict_moves_with_filter(
                input_df, self.model, self.feature_info, self.scaler,
                self.label_encoder, self.pokemon_valid_moves, top_k=15
            )
        except Exception as e:
            print(f"ERROR during prediction: {e}")
            import traceback
            traceback.print_exc()
            print("Choosing default random action due to prediction error.")
            return self.choose_random_move(battle)

        # 3. Process predictions and choose the best *valid* action
        chosen_action = None
        print("-" * 10 + " Predictions " + "-" * 10)
        if predictions_list and predictions_list[0] is not None and not predictions_list[0].empty:
            top_prediction_df = predictions_list[0]
            print(top_prediction_df[['rank', 'action_string', 'probability']].head().to_string(index=False))
            print("-" * (10 + 13 + 10)) # Adjust width if needed based on new col name length

            for rank_idx, row in top_prediction_df.iterrows():
                action_str = row['action_string'] # Read from the correct column
                action_str = f"move:{action_str.lower().replace(' ','')}"
                # --- END UPDATE ---
                prob = row['probability']
                action_parts = action_str.split(':', 1)
                # This condition should now work correctly as action_str has the colon
                print(action_parts)
                if len(action_parts) < 2:
                    print(f"  Rank {rank_idx+1}: Skipping malformed prediction '{action_str}'")
                    continue

                action_type = action_parts[0]
                # Line below is just a quick fix, forgot to mention to refactor ltr
                action_detail = action_parts[1].lower().replace(' ', '').replace('-', '')

                current_choice_valid = False
                if action_type == "move":
                    # --- Add Debug Print Here ---
                    print(f"  Rank {rank_idx+1}: Checking predicted move '{action_detail}'...")
                    # --- End Debug Print ---
                    for available_move in battle.available_moves:
                        # --- Add Detailed Comparison Debug Print Here ---
                        print(f"    Comparing prediction '{action_detail}' (Type: {type(action_detail)}) == available '{available_move.id}' (Type: {type(available_move.id)})")
                        # --- End Detailed Comparison Debug Print ---
                        if available_move.id == action_detail:
                            print(f"      MATCH FOUND!") # Confirmation
                            print(f"  Rank {rank_idx+1}: Predicted move '{available_move.id}' (Prob: {prob:.2%}) - VALID")
                            chosen_action = available_move
                            current_choice_valid = True
                            break # Found valid move, break inner loop

                if current_choice_valid:
                     print(f"  Action found and validated. Breaking from prediction loop.")
                     break # Break outer loop

        # 4. Handle cases where NO valid prediction was found
        if not chosen_action:
            chosen_action = battle.available_switches[0]
           

       # 5. Return the chosen action object
        # --- CORRECTED Final Logging ---
        final_action_log = "Unknown Action / Error"
        if chosen_action is None:
            final_action_log = "None (Error choosing default?)"
        elif isinstance(chosen_action, Move): # Specific type first
            final_action_log = chosen_action.id
        elif isinstance(chosen_action, Pokemon): # Specific type
            final_action_log = chosen_action.species
        elif isinstance(chosen_action, DefaultBattleOrder): # Specific type
            final_action_log = "Default Order"
        elif isinstance(chosen_action, BattleOrder): # Check base type LAST
            # Inspect the 'order' attribute
            if isinstance(chosen_action.order, Move):
                final_action_log = chosen_action.order.id
            elif isinstance(chosen_action.order, Pokemon):
                final_action_log = chosen_action.order.species
            else:
                final_action_log = str(chosen_action) # Use string representation
        # Add other BattleOrder types if necessary
        # --- END CORRECTED Final Logging ---

        print(f">>> Sending Action: {final_action_log}\n")

        # --- NEW LINE: Wrap the chosen action in a BattleOrder ---
        print(chosen_action)
        return self.create_order(chosen_action)



# ===========================================
# Main Execution Block
# ===========================================

async def main():
    """Sets up and runs the PredictionPlayer."""
    player = None
    try:
        server_config = poke_env.ShowdownServerConfiguration
        # Initialize player - this loads artifacts
        player = PredictionPlayer(
            battle_format=BATTLE_FORMAT,
            account_configuration=AccountConfiguration(SHOWDOWN_USERNAME, SHOWDOWN_PASSWORD),
            log_level=LOG_LEVEL,
            server_configuration=server_config,
            team=team,
            max_concurrent_battles=5,
            start_timer_on_battle_start=True
        )

        # Option 1: Accept challenges
        challenge_count = 50 # Number of challenges to accept
        print(f"\n{'-'*20}\n Bot ready. Accepting up to {challenge_count} challenge(s) as {SHOWDOWN_USERNAME} in format {BATTLE_FORMAT}...\n{'-'*20}")
        await player.accept_challenges(None, challenge_count) # Accept from anyone ('None')

        print("\nBot has finished its task(s).")

    except Exception as e:
        print(f"\nFATAL ERROR in main execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Optional: Graceful shutdown (poke-env might handle some of this)
        if player:
             print("Attempting to disconnect player...")
             # poke-env doesn't have an explicit disconnect method in recent versions
             # Closing the event loop usually handles it.
             pass


if __name__ == "__main__":
    print("Starting Poke-Env Bot...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot interrupted by user.")
    print("Bot script finished.")