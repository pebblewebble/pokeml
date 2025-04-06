import pandas as pd
import re
import copy
import os
import argparse
import glob

# --- Helper functions ---

def parse_hp(hp_str):
    """Parses HP string like '100/100' or 'fnt' into percentage."""
    if hp_str == 'fnt': return 0
    if hp_str.isdigit(): return int(hp_str)
    try:
        parts = hp_str.split('/')
        if len(parts) == 2: return round((int(parts[0]) / int(parts[1])) * 100) if int(parts[1]) > 0 else 0
        if hp_str.endswith('%'): return int(hp_str[:-1])
        return None
    except (ValueError, TypeError, IndexError, ZeroDivisionError): return None

def get_initial_pokemon_state():
    """Returns a default structure for a single Pokémon's state."""
    return {
        'species': 'Unknown',
        'hp_perc': 100, 'status': 'none', 'is_active': False, 'is_fainted': False,
        'boosts': {'atk': 0, 'def': 0, 'spa': 0, 'spd': 0, 'spe': 0},
        'terastallized': False, 'tera_type': 'none', 'revealed': False
    }

# --- Identify known conditions ---
PSEUDO_WEATHERS = {'Trick Room'}

# Includes hazards and non-hazards for parsing robustness
SIDE_CONDITIONS = {
    'Reflect', 'Light Screen', 'Aurora Veil', 'Safeguard', 'Tailwind',
 'Stealth Rock', 'Spikes', 'Toxic Spikes', 'Sticky Web',
}
HAZARD_CONDITIONS = {'Stealth Rock', 'Spikes', 'Toxic Spikes', 'Sticky Web'}

def get_initial_field_state():
    """Returns default field state including pseudo-weather and side conditions (updated)."""
    state = {
        'weather': 'none', 'weather_turns': 0,
        'pseudo_weather': 'none',
        'terrain': 'none', 'terrain_turns': 0,
        # Initialize based on the specific HAZARD_CONDITIONS set
        'p1_hazards': {hz: 0 for hz in HAZARD_CONDITIONS},
        'p2_hazards': {hz: 0 for hz in HAZARD_CONDITIONS},
        # Initialize based on conditions in RELEVANT_SIDE_CONDITIONS but NOT in HAZARD_CONDITIONS
        'p1_side': {sc: 0 for sc in SIDE_CONDITIONS if sc not in HAZARD_CONDITIONS},
        'p2_side': {sc: 0 for sc in SIDE_CONDITIONS if sc not in HAZARD_CONDITIONS},
    }
    return state

def parse_player_from_side(side_str):
    """Extracts 'p1' or 'p2' from side identifier like 'p1: www32'."""
    match = re.match(r'(p[12])', side_str)
    return match.group(1) if match else None

def parse_condition_name(condition_str):
    """Removes optional prefixes like 'move: '."""
    return condition_str.split(': ')[-1].strip()

def parse_pokemon_identifier(identifier):
    """Parses 'p1a: Ninetales' into ('p1', 'Ninetales')."""
    match = re.match(r'(p[12])[a-z]?(?::\s*(.*))?', identifier)
    if match:
        player = match.group(1)
        name_part = match.group(2).strip() if match.group(2) else None
        return player, name_part
    match_player_only = re.match(r'(p[12])$', identifier)
    if match_player_only: return match_player_only.group(1), None
    return None, None

# --- Main Parsing Function ---
def parse_showdown_replay(log_text, replay_id="unknown"):
    lines = log_text.strip().split('\n')
    game_states = []
    current_state = {
        'replay_id': replay_id, 'turn_number': 0, 'player_to_move': None,
        'p1': {}, 'p2': {}, 'field': get_initial_field_state(),
        'action_taken': None, 'battle_winner': None
    }
    player_names = {'p1': None, 'p2': None}
    slot_id_to_species = {'p1': {}, 'p2': {}}
    active_slot = {'p1': None, 'p2': None}
    current_teamsize = {'p1': 0, 'p2': 0}
    slot_counters = {'p1': 0, 'p2': 0}

    # --- Phase 1: Initial Setup ---
    for line in lines:
        parts = line.strip().split('|')
        if len(parts) < 2: continue
        event_type = parts[1]
        if event_type == 'player': 
            if len(parts) >= 4 and (player_names['p1'] == None or player_names['p2']==None): # Need at least |player|pX|name|...
                 player_id = parts[2]
                 name = parts[3]
                 if player_id in player_names: # Should be 'p1' or 'p2'
                      player_names[player_id] = name.strip() # Store the name and strip whitespace
                      print(f"DEBUG: Stored player name: {player_id} = '{player_names[player_id]}'") # Add debug print
                 else:
                      print(f"Warning: Invalid player_id '{player_id}' in |player| line: {line.strip()}")
            else:
                 print(f"Warning: Malformed |player| line (not enough parts): {line.strip()}")
        elif event_type == 'teamsize':
            player_id, size = parts[2], int(parts[3])
            current_teamsize[player_id] = size
            current_state[player_id] = {f'slot{i+1}': get_initial_pokemon_state() for i in range(size)}
        elif event_type == 'poke':
            player_id, details, *_ = parts[2:]
            species = details.split(',')[0].split('|')[-1].replace('-*', '')
            slot_counters[player_id] += 1
            slot_id = f'slot{slot_counters[player_id]}'
            if slot_counters[player_id] <= current_teamsize.get(player_id, 6):
                slot_id_to_species[player_id][slot_id] = species
                current_state[player_id][slot_id]['species'] = species

    def find_slot_id_by_species(player, species_name):
        # ... (keep the find_slot_id_by_species function as it was) ...
        if player not in slot_id_to_species: return None
        for slot_id, stored_species in slot_id_to_species[player].items():
            if stored_species == species_name: return slot_id
        base_species_name = species_name.split('-')[0]
        for slot_id, stored_species in slot_id_to_species[player].items():
             if stored_species.startswith(base_species_name): return slot_id
        print(f"ERROR: Could not find slot_id for {player} with species {species_name} in {slot_id_to_species.get(player)}")
        return None


    # --- Phase 2: Process Turns and Events ---
    last_action_player = None
    for line in lines:
        parts = line.strip().split('|')
        if len(parts) < 2: continue
        event_type = parts[1]

        # --- Basic Turn/Winner/Faint/Tera handling (mostly unchanged) ---
        if event_type == 'turn':
            current_state['turn_number'] = int(parts[2])
            last_action_player = 'p2' if current_state['turn_number'] > 0 else None
            current_state['player_to_move'] = 'p1'
        elif event_type == 'win':
             winner_name = parts[2]
             if player_names['p1'] == winner_name: current_state['battle_winner'] = 'p1'
             elif player_names['p2'] == winner_name: current_state['battle_winner'] = 'p2'
             else: current_state['battle_winner'] = 'unknown'
        elif event_type == 'faint':
            identifier = parts[2]
            player, _ = parse_pokemon_identifier(identifier)
            if not player: continue
            slot_id = active_slot.get(player)
            if slot_id and slot_id in current_state[player]: # Check slot exists
                 current_state[player][slot_id]['hp_perc'] = 0
                 current_state[player][slot_id]['status'] = 'fnt'
                 current_state[player][slot_id]['is_fainted'] = True
                 current_state[player][slot_id]['is_active'] = False
                 active_slot[player] = None
        elif event_type == '-terastallize':
             identifier, tera_type = parts[2], parts[3]
             player, _ = parse_pokemon_identifier(identifier)
             if not player: continue
             slot_id = active_slot.get(player)
             if slot_id and slot_id in current_state[player]:
                  current_state[player][slot_id]['terastallized'] = True
                  current_state[player][slot_id]['tera_type'] = tera_type

        # --- Core State Updates (HP, Status, Boosts) ---
        elif event_type in ['-damage', '-heal', '-sethp']:
            identifier, hp_str, *_ = parts[2:]
            player, _ = parse_pokemon_identifier(identifier)
            if not player: continue
            slot_id = active_slot.get(player)
            if slot_id and slot_id in current_state[player]:
                hp_perc = parse_hp(hp_str)
                if hp_perc is not None:
                     current_state[player][slot_id]['hp_perc'] = hp_perc
                     if hp_perc == 0 and current_state[player][slot_id]['status'] != 'fnt':
                          current_state[player][slot_id]['status'] = 'fnt'
                          current_state[player][slot_id]['is_fainted'] = True
                          current_state[player][slot_id]['is_active'] = False
        elif event_type == '-status':
            identifier, status, *_ = parts[2:]
            player, _ = parse_pokemon_identifier(identifier)
            if not player: continue
            slot_id = active_slot.get(player)
            if slot_id and slot_id in current_state[player]: current_state[player][slot_id]['status'] = status
        elif event_type == '-curestatus':
            identifier, status, *_ = parts[2:]
            player, _ = parse_pokemon_identifier(identifier)
            if not player: continue
            slot_id = active_slot.get(player)
            if slot_id and slot_id in current_state[player] and current_state[player][slot_id]['status'] == status:
                current_state[player][slot_id]['status'] = 'none'
        elif event_type == '-boost' or event_type == '-unboost':
            identifier, stat, amount, *_ = parts[2:]
            player, _ = parse_pokemon_identifier(identifier)
            if not player: continue
            slot_id = active_slot.get(player)
            try:
                stat_key = stat.lower()
                amount_val = int(amount)
                if slot_id and slot_id in current_state[player] and stat_key in current_state[player][slot_id]['boosts']:
                    mult = 1 if event_type == '-boost' else -1
                    boosts = current_state[player][slot_id]['boosts']
                    boosts[stat_key] = max(-6, min(6, boosts[stat_key] + (mult * amount_val)))
            except (ValueError, KeyError): pass

        # --- Switch / Drag ---
        elif event_type in ['switch', 'drag']:
            identifier, details, hp_str = parts[2], parts[3], parts[4]
            player, _ = parse_pokemon_identifier(identifier)
            if not player: continue
            incoming_species = details.split(',')[0].replace('-*', '')
            incoming_slot_id = find_slot_id_by_species(player, incoming_species)
            if not incoming_slot_id: continue

            old_active_slot_id = active_slot.get(player)
            if old_active_slot_id and old_active_slot_id in current_state[player]:
                 current_state[player][old_active_slot_id]['is_active'] = False
            active_slot[player] = incoming_slot_id

            hp_perc = parse_hp(hp_str) or 100 # Default to 100 if parse fails
            poke_state = current_state[player][incoming_slot_id]
            poke_state['species'] = incoming_species
            poke_state['hp_perc'] = hp_perc
            poke_state['is_active'] = True
            poke_state['is_fainted'] = (hp_perc == 0)
            poke_state['revealed'] = True
            poke_state['boosts'] = {k: 0 for k in poke_state['boosts']}
            poke_state['terastallized'] = False
            poke_state['tera_type'] = 'none'

        # --- Field State Updates ---
        elif event_type == '-weather':
             current_state['field']['weather'] = parts[2].split(' ')[0]
        elif event_type == '-terrain':
             current_state['field']['terrain'] = parts[2].split(' ')[0]
        elif event_type == '-fieldstart':
             condition = parse_condition_name(parts[2])
             if condition in PSEUDO_WEATHERS: current_state['field']['pseudo_weather'] = condition
        elif event_type == '-fieldend':
             condition = parse_condition_name(parts[2])
             if current_state['field']['pseudo_weather'] == condition: current_state['field']['pseudo_weather'] = 'none'

        # --- Side Condition / Hazard Updates (Corrected Logic) ---
        elif event_type == '-sidestart':
             if len(parts) < 4: continue # Ensure enough parts
             side_identifier, condition_str = parts[2], parts[3]
             affected_side_player = parse_player_from_side(side_identifier) # p1 or p2
             condition = parse_condition_name(condition_str)
             if not affected_side_player or not condition: continue # Skip if parsing failed

             # Update Hazards (on the side specified in the log)
             if condition in HAZARD_CONDITIONS:
                 target_hazard_key = f'{affected_side_player}_hazards'
                 if condition in current_state['field'][target_hazard_key]:
                     hazard_dict = current_state['field'][target_hazard_key]
                     max_layers = 3 if condition == 'Spikes' else (2 if condition == 'Toxic Spikes' else 1)
                     hazard_dict[condition] = min(max_layers, hazard_dict[condition] + 1)
                     # print(f"DEBUG SIDESTART: Hazard '{condition}' set on {affected_side_player}. State: {hazard_dict}") # Optional debug
             # Update Non-Hazard Side Conditions (on the side specified in the log)
             elif condition in SIDE_CONDITIONS:
                  target_side_key = f'{affected_side_player}_side'
                  if condition in current_state['field'][target_side_key]:
                       current_state['field'][target_side_key][condition] = 1 # Mark active
                       # print(f"DEBUG SIDESTART: Side Condition '{condition}' started for {affected_side_player}.") # Optional debug

        elif event_type == '-sideend':
            if len(parts) < 4: continue
            side_identifier, condition_str = parts[2], parts[3]
            affected_side_player = parse_player_from_side(side_identifier)
            condition = parse_condition_name(condition_str)
            if not affected_side_player or not condition: continue

            # End Non-Hazard Side Conditions
            if condition in SIDE_CONDITIONS and condition not in HAZARD_CONDITIONS:
                 target_side_key = f'{affected_side_player}_side'
                 if condition in current_state['field'][target_side_key]:
                      current_state['field'][target_side_key][condition] = 0 # Mark inactive
                      # print(f"DEBUG SIDEEND: Side Condition '{condition}' ended for {affected_side_player}.") # Optional debug
            # Note: Hazards are typically removed by moves (Defog, etc.), not -sideend

        # --- Action Recording ---
        is_player_action = False
        action_description = None
        acting_player = None
        if event_type == 'move':
            identifier, move_name, *_ = parts[2:]
            player, _ = parse_pokemon_identifier(identifier)
            if player: acting_player, is_player_action, action_description = player, True, f"move:{move_name}"
        elif event_type == 'switch' and len(parts) > 3 and parts[1] == 'switch':
            identifier, details, *_ = parts[2:]
            player, _ = parse_pokemon_identifier(identifier)
            species = details.split(',')[0].replace('-*', '')
            if player: acting_player, is_player_action, action_description = player, True, f"switch:{species}"

        if is_player_action and acting_player:
            state_snapshot = copy.deepcopy(current_state)
            state_snapshot['action_taken'] = action_description
            state_snapshot['player_to_move'] = acting_player
            game_states.append(state_snapshot)
            last_action_player = acting_player
            current_state['player_to_move'] = 'p2' if acting_player == 'p1' else 'p1'

    # --- Finalize ---
    final_winner = current_state.get('battle_winner')
    for state in game_states: state['battle_winner'] = final_winner
    return game_states

# --- Flattening function (No changes needed from previous refinement) ---
def flatten_state(state_dict):
    flat = {}
    flat['replay_id'] = state_dict['replay_id']
    flat['turn_number'] = state_dict['turn_number']
    flat['player_to_move'] = state_dict['player_to_move']
    flat['action_taken'] = state_dict['action_taken']
    flat['battle_winner'] = state_dict['battle_winner']

    # Pokemon States
    for player in ['p1', 'p2']:
        if player in state_dict:
             num_slots = len(state_dict[player])
             for i in range(num_slots):
                 slot_id = f'slot{i+1}'
                 if slot_id in state_dict[player]:
                     poke_state = state_dict[player][slot_id]
                     prefix = f'{player}_{slot_id}'
                     flat[f'{prefix}_species'] = poke_state.get('species', 'Unknown')
                     flat[f'{prefix}_hp_perc'] = poke_state.get('hp_perc')
                     flat[f'{prefix}_status'] = poke_state.get('status', 'none')
                     flat[f'{prefix}_is_active'] = int(poke_state.get('is_active', False))
                     flat[f'{prefix}_is_fainted'] = int(poke_state.get('is_fainted', False))
                     flat[f'{prefix}_terastallized'] = int(poke_state.get('terastallized', False))
                     flat[f'{prefix}_tera_type'] = poke_state.get('tera_type', 'none')
                     boosts = poke_state.get('boosts', {})
                     for stat in ['atk', 'def', 'spa', 'spd', 'spe']:
                         flat[f'{prefix}_boost_{stat}'] = boosts.get(stat, 0)

    # Field States
    if 'field' in state_dict:
        field_state = state_dict['field']
        flat['field_weather'] = field_state.get('weather', 'none')
        flat['field_terrain'] = field_state.get('terrain', 'none')
        flat['field_pseudo_weather'] = field_state.get('pseudo_weather', 'none')

        # Hazards (Correctly keyed by the side they are ON)
        for affected_side, hazard_key in [('p1', 'p1_hazards'), ('p2', 'p2_hazards')]:
            hazard_dict = field_state.get(hazard_key, {})
            for hazard, value in hazard_dict.items():
                 flat_key = f'{affected_side}_hazard_{hazard.lower().replace(" ", "_")}'
                 flat[flat_key] = value

        # Side Conditions (Correctly keyed by the side they are ON)
        for affected_side, side_key in [('p1', 'p1_side'), ('p2', 'p2_side')]:
            side_dict = field_state.get(side_key, {})
            for condition, value in side_dict.items():
                 flat_key = f'{affected_side}_side_{condition.lower().replace(" ", "_")}'
                 flat[flat_key] = value

    return flat


# --- Main Script Logic (Add --limit for testing) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse Pokemon Showdown replay log files into a single data table (Parquet/CSV).")
    parser.add_argument("input_dir", type=str, help="Directory containing the downloaded .log files.")
    parser.add_argument("output_file", type=str, help="Path to save the output data table (e.g., output.parquet or output.csv).")
    parser.add_argument("--format", choices=['parquet', 'csv'], default='parquet', help="Output file format (parquet or csv). Parquet recommended.")
    parser.add_argument("--limit", type=int, default=None, help="Optional: Limit the number of files to process (for testing).")

    args = parser.parse_args()
    # ... (rest of the main block remains the same: find files, loop, parse, aggregate, save) ...
    log_files = glob.glob(os.path.join(args.input_dir, '*.log'))
    if not log_files: print(f"Error: No .log files found in directory: {args.input_dir}"); exit(1)
    if args.limit: print(f"Limiting processing to the first {args.limit} files found."); log_files = log_files[:args.limit]
    print(f"Found {len(log_files)} .log files to parse in '{args.input_dir}'.")
    all_flattened_states = []; parsed_files = 0; failed_files = 0
    for i, file_path in enumerate(log_files):
        filename = os.path.basename(file_path)
        print(f"Parsing file {i+1}/{len(log_files)}: {filename} ...", end=' ', flush=True)
        match = re.search(r'-(\d+)\.log$', filename); replay_id = match.group(1) if match else f"unknown_{filename}"
        try:
            with open(file_path, 'r', encoding='utf-8') as f: log_content = f.read()
            parsed_replay_states = parse_showdown_replay(log_content, replay_id=replay_id)
            if not parsed_replay_states: print("Warning: No states extracted."); continue
            flattened_replay_states = [flatten_state(s) for s in parsed_replay_states]
            all_flattened_states.extend(flattened_replay_states); print(f"OK ({len(flattened_replay_states)} states added)."); parsed_files += 1
        except FileNotFoundError: print(f"Error: File not found: {filename}"); failed_files += 1
        except Exception as e: print(f"FAIL. Error: {e}"); failed_files += 1 #; import traceback; traceback.print_exc() # Uncomment for debug
    print(f"\nParsing complete.\nSuccessfully parsed: {parsed_files} files\nFailed to parse:   {failed_files} files\nTotal states collected: {len(all_flattened_states)}")
    if not all_flattened_states: print("No states collected, skipping output."); exit(0)
    print("\nCreating DataFrame..."); df = pd.DataFrame(all_flattened_states); print("DataFrame created successfully.")
    print(f"Saving DataFrame to '{args.output_file}' (format: {args.format})...")
    try:
        if args.format == 'parquet': df.to_parquet(args.output_file, index=False, engine='pyarrow')
        elif args.format == 'csv': df.to_csv(args.output_file, index=False)
        print("DataFrame saved successfully.\n\nFinal DataFrame Info:"); df.info(verbose=True, show_counts=True)
    except Exception as e: print(f"\nError creating or saving DataFrame: {e}"); import traceback; traceback.print_exc(); exit(1)