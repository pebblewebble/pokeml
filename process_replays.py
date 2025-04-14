# --- START OF (COMPLETE Modified) process_replays.py ---

import pandas as pd
import re
import copy
import os
import argparse
import glob
import json # For potentially cleaner representation if needed, though using sorted string

# --- Helper functions ---
def parse_hp(hp_str):
    """Parses Showdown HP strings (e.g., '100/100', '50%', 'fnt', '75', '36/100 tox') into percentages."""
    if hp_str == 'fnt': return 0
    if hp_str.isdigit(): return int(hp_str)

    try:
        # --- FIX: Attempt to strip status BEFORE splitting ---
        # Match HP part (X/Y or X%) potentially followed by space and status word (like 'tox', 'brn')
        match_hp_with_status = re.match(r'^(\d+/\d+|\d+%?)(\s+\w+)?$', hp_str.strip())
        if match_hp_with_status:
            # If matched, use only the HP part for further parsing
            hp_part = match_hp_with_status.group(1)
            # Now parse the clean hp_part
            if '/' in hp_part:
                parts = hp_part.split('/')
                if len(parts) == 2:
                    # Ensure denominator is not zero before division
                    denominator = int(parts[1])
                    return round((int(parts[0]) / denominator) * 100) if denominator > 0 else 0
            elif hp_part.endswith('%'):
                return int(hp_part[:-1])
            elif hp_part.isdigit(): # Should be caught earlier, but good failsafe
                 return int(hp_part)
            else:
                 # The regex matched but the captured hp_part is invalid? Should not happen with this regex.
                 print(f"Warning: Regex matched but failed to parse HP part: '{hp_part}' from '{hp_str}'")
                 return None
        # --- END FIX ---

        # --- Original checks (if regex didn't match, meaning no status or different format) ---
        # This path is now less likely to be needed but kept for safety/other formats
        elif '/' in hp_str: # Check if it looks like fraction even if regex failed
             parts = hp_str.split('/')
             if len(parts) == 2:
                  denominator = int(parts[1])
                  return round((int(parts[0]) / denominator) * 100) if denominator > 0 else 0
        elif hp_str.endswith('%'):
             return int(hp_str[:-1])

        # If none of the above worked
        return None # Return None if format is unrecognized

    except (ValueError, TypeError, IndexError, ZeroDivisionError) as e:
        # Print a warning for unexpected errors during parsing attempts
        # Avoid printing too much for common non-matches
        # print(f"Warning: Exception during HP parsing '{hp_str}': {e}")
        return None # Indicate failure to parse

def get_initial_pokemon_state():
    """Returns a dictionary representing the default state of a Pokémon slot."""
    return {
        'species': 'Unknown',
        'hp_perc': 100,
        'status': 'none',
        'is_active': False,
        'is_fainted': False,
        'boosts': {'atk': 0, 'def': 0, 'spa': 0, 'spd': 0, 'spe': 0},
        'terastallized': False,
        'tera_type': 'none',
        'revealed': False, # Has this Pokemon been seen on the field yet?
        'revealed_moves': set() # Set of move names used by this Pokemon slot
    }

# --- Identify known conditions ---
PSEUDO_WEATHERS = {'Trick Room'} # Field conditions affecting both sides
SIDE_CONDITIONS = { # Conditions affecting only one side
    'Reflect', 'Light Screen', 'Aurora Veil', 'Safeguard', 'Tailwind',
    'Stealth Rock', 'Spikes', 'Toxic Spikes', 'Sticky Web',
}
HAZARD_CONDITIONS = {'Stealth Rock', 'Spikes', 'Toxic Spikes', 'Sticky Web'} # Subset of side conditions that are hazards

def get_initial_field_state():
    """Returns a dictionary representing the default state of the battle field."""
    state = {
        'weather': 'none', 'weather_turns': 0, # Weather condition and potential duration
        'pseudo_weather': 'none', # e.g., Trick Room
        'terrain': 'none', 'terrain_turns': 0, # Terrain condition and potential duration
        'p1_hazards': {hz: 0 for hz in HAZARD_CONDITIONS}, # Hazard layers for player 1
        'p2_hazards': {hz: 0 for hz in HAZARD_CONDITIONS}, # Hazard layers for player 2
        'p1_side': {sc: 0 for sc in SIDE_CONDITIONS if sc not in HAZARD_CONDITIONS}, # Other side conditions for p1 (0=inactive, 1=active, potentially turns later)
        'p2_side': {sc: 0 for sc in SIDE_CONDITIONS if sc not in HAZARD_CONDITIONS}, # Other side conditions for p2
    }
    return state

def parse_player_from_side(side_str):
    """Extracts player identifier ('p1' or 'p2') from a side identifier string (e.g., 'p1: Garchomp')."""
    match = re.match(r'(p[12])', side_str)
    return match.group(1) if match else None

def parse_condition_name(condition_str):
    """Extracts the condition name from a Showdown condition string (e.g., 'move: Stealth Rock')."""
    # Handles formats like 'move: Stealth Rock', 'ability: Drought', 'item: Leftovers', or just 'Stealth Rock'
    return condition_str.split(': ')[-1].strip()

def parse_pokemon_identifier(identifier):
    """
    Extracts player ('p1'/'p2') and nickname/species name from a Pokémon identifier string.
    Examples: 'p1a: Garchomp', 'p2b: Pikachu', 'p1: Landorus', 'p2a'
    Returns (player, name_part or None).
    """
    match = re.match(r'(p[12])[a-z]?(?::\s*(.*))?', identifier)
    if match:
        player = match.group(1)
        # Clean the name part slightly (remove potential leading/trailing whitespace)
        name_part = match.group(2).strip() if match.group(2) else None
        return player, name_part
    # Handle cases like just 'p1' or 'p2' if they occur (e.g., in side conditions)
    match_player_only = re.match(r'(p[12])$', identifier)
    if match_player_only:
        return match_player_only.group(1), None
    return None, None

# --- Main Parsing Function ---
def parse_showdown_replay(log_text, replay_id="unknown"):
    """
    Parses a raw Pokemon Showdown replay log text into a list of game state dictionaries.
    Each dictionary represents the state *before* a player action (move or switch).
    """
    lines = log_text.strip().split('\n')
    game_states = [] # List to store state snapshots before each action
    current_state = {
        'replay_id': replay_id,
        'turn_number': 0,
        'player_to_move': None, # Which player is about to take an action in this state
        'p1': {}, # Dictionary mapping 'slotX' to pokemon state dict
        'p2': {}, # Dictionary mapping 'slotX' to pokemon state dict
        'field': get_initial_field_state(),
        'action_taken': None, # The action taken *after* this state (for labeling)
        'battle_winner': None, # Final winner ('p1', 'p2', 'unknown', etc.)
        'last_move_p1': 'none', # Last move used by p1 *before* this state snapshot
        'last_move_p2': 'none'  # Last move used by p2 *before* this state snapshot
    }
    player_names = {'p1': None, 'p2': None} # Stores player usernames
    # Maps slotX (e.g., 'slot1') to the initial species revealed in |poke| lines
    slot_id_to_species = {'p1': {}, 'p2': {}}
    # Maps nickname encountered dynamically to the corresponding slot_id
    nickname_to_slot_id = {'p1': {}, 'p2': {}}
    # Tracks the currently active slot_id for each player
    active_slot = {'p1': None, 'p2': None}
    # Tracks the team size specified at the beginning
    current_teamsize = {'p1': 0, 'p2': 0}
    # Helper counters to assign initial slot IDs during |poke| processing
    slot_counters = {'p1': 0, 'p2': 0}
    # Tracks the most recent move used by each player globally (for state snapshotting)
    last_move_seen = {'p1': 'none', 'p2': 'none'}

    # --- Phase 1: Initial Setup (Teamsize, Initial Pokemon Species) ---
    for line in lines:
        parts = line.strip().split('|')
        if len(parts) < 2: continue
        event_type = parts[1]

        if event_type == 'player':
             # Store player names if available and not already set
             if len(parts) >= 4 and (player_names['p1'] is None or player_names['p2'] is None):
                  player_id = parts[2]
                  name = parts[3]
                  if player_id in player_names:
                       player_names[player_id] = name.strip()
                  else:
                       # Only warn if player ID itself is invalid
                       print(f"Warning: Invalid player_id '{player_id}' in |player| line: {line.strip()}")
             elif player_names['p1'] is None or player_names['p2'] is None:
                  # Warn about malformed line only if names aren't already found
                  print(f"Warning: Malformed |player| line (and names not set): {line.strip()}")

        elif event_type == 'teamsize':
            # Initialize player state dictionary structure based on team size
            if len(parts) >= 4:
                player_id, size_str = parts[2], parts[3]
                try:
                    size = int(size_str)
                    if player_id in current_teamsize:
                        current_teamsize[player_id] = size
                        # Initialize the player's slots in the current state
                        current_state[player_id] = {f'slot{i+1}': get_initial_pokemon_state() for i in range(size)}
                    else:
                        print(f"Warning: Invalid player_id '{player_id}' in |teamsize| line: {line.strip()}")
                except ValueError:
                    print(f"Warning: Invalid teamsize value '{size_str}' in |teamsize| line: {line.strip()}")
            else:
                print(f"Warning: Malformed |teamsize| line: {line.strip()}")


        elif event_type == 'poke':
            # Store the initial species for each slot ID
            if len(parts) >= 3:
                player_id, details, *_ = parts[2:]
                if player_id in slot_counters:
                    species = details.split(',')[0].split('|')[-1].replace('-*', '') # Basic species name extraction
                    slot_counters[player_id] += 1
                    slot_id = f'slot{slot_counters[player_id]}'
                    # Check if slot counter exceeds reported team size (can happen in some formats?)
                    if slot_counters[player_id] <= current_teamsize.get(player_id, 6): # Default to 6 if teamsize unknown
                        if player_id in current_state and slot_id in current_state[player_id]:
                            # Store initial species for lookup later
                            slot_id_to_species[player_id][slot_id] = species
                            # Update the initial state for this slot
                            current_state[player_id][slot_id]['species'] = species
                        else:
                             # This might happen if |poke| lines appear before |teamsize| or if size is wrong
                             print(f"Warning: Slot '{slot_id}' referenced in |poke| for {player_id} but not initialized (teamsize issue?). Line: {line.strip()}")
                             # Attempt recovery: Initialize the slot if player state exists
                             if player_id in current_state:
                                  current_state[player_id][slot_id] = get_initial_pokemon_state()
                                  current_state[player_id][slot_id]['species'] = species
                                  slot_id_to_species[player_id][slot_id] = species # Also update lookup map
                else:
                     print(f"Warning: Invalid player_id '{player_id}' in |poke| line: {line.strip()}")
            else:
                print(f"Warning: Malformed |poke| line: {line.strip()}")


    # --- Slot ID Lookup Function ---
    def find_slot_id(player, identifier_name, species_from_details):
        """
        Finds the slot_id for a Pokémon based on identifier name (nickname/species) and species from details.
        Updates the nickname mapping dynamically.
        Prioritizes: Nickname map -> Exact species match -> Base species match. Returns None on failure.
        """
        nonlocal nickname_to_slot_id # Allow modification of the outer scope variable

        if not player or player not in slot_id_to_species: return None # Basic checks

        # 1. Check Nickname Mapping
        if identifier_name and player in nickname_to_slot_id and identifier_name in nickname_to_slot_id[player]:
            found_slot_id = nickname_to_slot_id[player][identifier_name]
            # Basic validation: does the slot still exist and roughly match the expected species?
            if found_slot_id in slot_id_to_species.get(player, {}):
                 initial_species = slot_id_to_species[player][found_slot_id]
                 # Check if species match (exactly or base form)
                 if initial_species == species_from_details or initial_species.startswith(species_from_details.split('-')[0]):
                      return found_slot_id
                 # else: Nickname might be stale (e.g., Ditto transformed), continue to species lookup

        # 2. Check Initial Species Mapping (using species_from_details)
        target_species = species_from_details
        for slot_id, initial_species in slot_id_to_species[player].items():
            if initial_species == target_species:
                # Found match based on species. Update nickname map if applicable.
                if identifier_name and identifier_name != target_species:
                    # Avoid mapping if nickname is already mapped to a *different* slot
                    if identifier_name not in nickname_to_slot_id.get(player, {}) or nickname_to_slot_id[player].get(identifier_name) == slot_id:
                        nickname_to_slot_id.setdefault(player, {})[identifier_name] = slot_id
                        # print(f"DEBUG: Mapped nickname '{identifier_name}' to slot '{slot_id}' (player {player}) via species match.") # Optional Debug
                return slot_id

        # 3. Check Base Species Mapping
        base_target_species = target_species.split('-')[0]
        for slot_id, initial_species in slot_id_to_species[player].items():
             if initial_species.startswith(base_target_species):
                 # Found match based on base species. Update nickname map if applicable.
                 if identifier_name and identifier_name != target_species:
                     if identifier_name not in nickname_to_slot_id.get(player, {}) or nickname_to_slot_id[player].get(identifier_name) == slot_id:
                          nickname_to_slot_id.setdefault(player, {})[identifier_name] = slot_id
                          # print(f"DEBUG: Mapped nickname '{identifier_name}' to slot '{slot_id}' (player {player}) via base species match.") # Optional Debug
                 return slot_id

        # 4. Error Case - Lookup Failed
        print(f"ERROR: Could not find slot_id for {player} using nickname='{identifier_name}' OR species='{species_from_details}'.")
        print(f"  Initial Species Map: {slot_id_to_species.get(player)}")
        print(f"  Known Nicknames: {nickname_to_slot_id.get(player)}")
        # Avoid printing entire state for brevity in errors
        # print(f"  Current State Species: {{s: p.get('species') for s, p in current_state.get(player, {}).items()}}")
        print(f"  Active Slot: {active_slot.get(player)}")
        return None # Return None to indicate failure

    # --- Phase 2: Process Turns and Events ---
    last_action_player = None
    for line_num, line in enumerate(lines): # Add line number for debugging
        parts = line.strip().split('|')
        if len(parts) < 2: continue
        event_type = parts[1]

        try: # Add try-except block around event processing for better error isolation
            # --- Basic Turn/Winner/Faint/Tera handling ---
            if event_type == 'turn':
                current_state['turn_number'] = int(parts[2])
                # Reset who might move first (simplistic)
                last_action_player = 'p2' if current_state['turn_number'] > 0 else None
                current_state['player_to_move'] = 'p1' # Assume p1 starts turn request phase

            elif event_type == 'win':
                 winner_name = parts[2]
                 # Determine winner based on stored player names
                 if player_names['p1'] and player_names['p2']:
                     if player_names['p1'] == winner_name: current_state['battle_winner'] = 'p1'
                     elif player_names['p2'] == winner_name: current_state['battle_winner'] = 'p2'
                     else: current_state['battle_winner'] = 'unknown' # Winner name doesn't match known players
                 else:
                     # Player names weren't parsed correctly
                     print(f"Warning: Could not determine winner for name '{winner_name}' as player names not fully parsed ({player_names})")
                     current_state['battle_winner'] = 'unknown_players_not_parsed'

            elif event_type == 'faint':
                if len(parts) >= 3:
                    identifier = parts[2]
                    player, _ = parse_pokemon_identifier(identifier)
                    if player:
                        slot_id = active_slot.get(player)
                        if slot_id and player in current_state and slot_id in current_state[player]:
                             poke_state = current_state[player].get(slot_id)
                             if poke_state:
                                  poke_state['hp_perc'] = 0
                                  poke_state['status'] = 'fnt'
                                  poke_state['is_fainted'] = True
                                  poke_state['is_active'] = False
                             active_slot[player] = None # Pokemon is no longer active
                        # else: # Debugging if needed
                        #      print(f"Warning: Fainted pokemon slot not found/active? Player: {player}, Slot: {slot_id}, Active: {active_slot}, StateKeys: {current_state.get(player, {}).keys()}. Line: {line.strip()}")
                else: print(f"Warning: Malformed |faint| line: {line.strip()}")


            elif event_type == '-terastallize':
                 if len(parts) >= 4:
                     identifier, tera_type = parts[2], parts[3]
                     player, _ = parse_pokemon_identifier(identifier)
                     if player:
                         slot_id = active_slot.get(player)
                         if slot_id and player in current_state and slot_id in current_state[player]:
                              poke_state = current_state[player].get(slot_id)
                              if poke_state:
                                poke_state['terastallized'] = True
                                poke_state['tera_type'] = tera_type
                 else: print(f"Warning: Malformed |-terastallize| line: {line.strip()}")

            # --- Core State Updates (HP, Status, Boosts) ---
            elif event_type in ['-damage', '-heal', '-sethp']:
                if len(parts) >= 4:
                    identifier, hp_str, *_ = parts[2:]
                    player, _ = parse_pokemon_identifier(identifier)
                    if player:
                        slot_id = active_slot.get(player)
                        if slot_id and player in current_state and slot_id in current_state[player]:
                            poke_state = current_state[player].get(slot_id)
                            if poke_state:
                                hp_perc = parse_hp(hp_str)
                                if hp_perc is not None:
                                     poke_state['hp_perc'] = hp_perc
                                     # Update faint status if HP reaches 0 from damage/sethp
                                     if hp_perc == 0 and poke_state['status'] != 'fnt':
                                          poke_state['status'] = 'fnt'
                                          poke_state['is_fainted'] = True
                                          poke_state['is_active'] = False # Should already be handled by faint event, but good failsafe
                                elif '[from]' not in hp_str and 'of' not in hp_str: # Avoid warnings for effect sources
                                     print(f"Warning: Could not parse HP value '{hp_str}' in {event_type} line: {line.strip()}")
                else: print(f"Warning: Malformed {event_type} line: {line.strip()}")


            elif event_type == '-status':
                if len(parts) >= 4:
                    identifier, status, *_ = parts[2:]
                    player, _ = parse_pokemon_identifier(identifier)
                    if player:
                        slot_id = active_slot.get(player)
                        if slot_id and player in current_state and slot_id in current_state[player]:
                             poke_state = current_state[player].get(slot_id)
                             if poke_state: poke_state['status'] = status
                else: print(f"Warning: Malformed |-status| line: {line.strip()}")

            elif event_type == '-curestatus':
                if len(parts) >= 4:
                    identifier, status, *_ = parts[2:]
                    player, _ = parse_pokemon_identifier(identifier)
                    if player:
                        slot_id = active_slot.get(player)
                        if slot_id and player in current_state and slot_id in current_state[player]:
                             poke_state = current_state[player].get(slot_id)
                             if poke_state and poke_state['status'] == status:
                                  poke_state['status'] = 'none'
                else: print(f"Warning: Malformed |-curestatus| line: {line.strip()}")

            elif event_type == '-boost' or event_type == '-unboost':
                if len(parts) >= 5:
                    identifier, stat, amount_str, *_ = parts[2:]
                    player, _ = parse_pokemon_identifier(identifier)
                    if player:
                        slot_id = active_slot.get(player)
                        if slot_id and player in current_state and slot_id in current_state[player]:
                             poke_state = current_state[player].get(slot_id)
                             if poke_state and 'boosts' in poke_state:
                                 try:
                                     stat_key = stat.lower()
                                     amount_val = int(amount_str)
                                     if stat_key in poke_state['boosts']:
                                         mult = 1 if event_type == '-boost' else -1
                                         boosts = poke_state['boosts']
                                         boosts[stat_key] = max(-6, min(6, boosts[stat_key] + (mult * amount_val)))
                                 except (ValueError, KeyError):
                                     print(f"Warning: Invalid stat/amount '{stat}/{amount_str}' in {event_type} line: {line.strip()}")
                else: print(f"Warning: Malformed {event_type} line: {line.strip()}")


            # --- Switch / Drag ---
            elif event_type in ['switch', 'drag']:
                if len(parts) >= 5:
                    identifier, details, hp_str, *_ = parts[2:5] # Ensure we get at least these parts
                    player, ident_name_part = parse_pokemon_identifier(identifier)
                    if player:
                        incoming_species = details.split(',')[0].replace('-*', '') # Get species from details string

                        # Find the slot ID using the centralized function
                        incoming_slot_id = find_slot_id(player, ident_name_part, incoming_species)

                        if incoming_slot_id:
                            # Deactivate old Pokémon (if any)
                            old_active_slot_id = active_slot.get(player)
                            if old_active_slot_id and player in current_state and old_active_slot_id in current_state[player]:
                                 old_poke_state = current_state[player].get(old_active_slot_id)
                                 if old_poke_state: old_poke_state['is_active'] = False

                            # Activate new Pokémon
                            active_slot[player] = incoming_slot_id

                            hp_perc = parse_hp(hp_str)
                            if hp_perc is None:
                                print(f"Warning: Could not parse HP '{hp_str}' on switch/drag. Defaulting to 100. Line: {line.strip()}")
                                hp_perc = 100 # Default to 100 if parse fails

                            # Access the state for the incoming Pokémon (should exist if slot found)
                            if incoming_slot_id in current_state.get(player, {}):
                                poke_state = current_state[player][incoming_slot_id]
                                poke_state['species'] = incoming_species # Update species (could change from base form)
                                poke_state['hp_perc'] = hp_perc
                                poke_state['is_active'] = True
                                poke_state['is_fainted'] = (hp_perc == 0)
                                poke_state['revealed'] = True # Mark as seen
                                # Reset temporary conditions on switch IN
                                poke_state['boosts'] = {k: 0 for k in poke_state.get('boosts', get_initial_pokemon_state()['boosts'])}
                                poke_state['terastallized'] = False
                                poke_state['tera_type'] = 'none'
                                # Revealed moves are persistent, not reset
                            else:
                                # This case should ideally not happen if find_slot_id worked correctly
                                print(f"Critical Error: Slot '{incoming_slot_id}' found but not in current_state for {player} during switch. Line: {line.strip()}")
                        else:
                            # Error message already printed by find_slot_id
                            print(f"Warning: Skipping switch/drag update due to failed slot lookup. Line: {line.strip()}")
                else: print(f"Warning: Malformed |{event_type}| line: {line.strip()}")

            # --- Field State Updates ---
            elif event_type == '-weather':
                 # Examples: |-weather|SunnyDay|, |-weather|RainDance|[from] ability: Drizzle|, |-weather|none|
                 weather_state = parts[2].split('[from]')[0].split('[upkeep]')[0].strip()
                 current_state['field']['weather'] = weather_state
                 # TODO: Handle weather duration if needed

            elif event_type == '-terrain':
                 # Examples: |-terrain|Electric Terrain|, |-terrain|none|
                 terrain_state = parts[2].split('[from]')[0].strip()
                 current_state['field']['terrain'] = terrain_state
                 # TODO: Handle terrain duration if needed

            elif event_type == '-fieldstart':
                 # Example: |-fieldstart|move: Trick Room|
                 if len(parts) >= 3:
                     condition = parse_condition_name(parts[2])
                     if condition in PSEUDO_WEATHERS:
                         current_state['field']['pseudo_weather'] = condition
                         # TODO: Handle pseudo-weather duration if needed
                 else: print(f"Warning: Malformed |-fieldstart| line: {line.strip()}")

            elif event_type == '-fieldend':
                 # Example: |-fieldend|move: Trick Room|
                 if len(parts) >= 3:
                     condition = parse_condition_name(parts[2])
                     if current_state['field']['pseudo_weather'] == condition:
                         current_state['field']['pseudo_weather'] = 'none'
                 else: print(f"Warning: Malformed |-fieldend| line: {line.strip()}")


            # --- Side Condition / Hazard Updates ---
            elif event_type == '-sidestart':
                 # Example: |-sidestart|p1: PlayerName|move: Stealth Rock|
                 if len(parts) >= 4:
                     side_identifier, condition_str = parts[2], parts[3]
                     affected_side_player = parse_player_from_side(side_identifier)
                     condition = parse_condition_name(condition_str)

                     if affected_side_player and condition:
                         target_hazard_key = f'{affected_side_player}_hazards'
                         target_side_key = f'{affected_side_player}_side'

                         # Handle Hazards
                         if condition in HAZARD_CONDITIONS and target_hazard_key in current_state['field']:
                             hazard_dict = current_state['field'][target_hazard_key]
                             if condition in hazard_dict:
                                 max_layers = 3 if condition == 'Spikes' else (2 if condition == 'Toxic Spikes' else 1)
                                 hazard_dict[condition] = min(max_layers, hazard_dict[condition] + 1)
                         # Handle Other Side Conditions
                         elif condition in SIDE_CONDITIONS and condition not in HAZARD_CONDITIONS and target_side_key in current_state['field']:
                              side_dict = current_state['field'][target_side_key]
                              if condition in side_dict:
                                   side_dict[condition] = 1 # Mark as active (usually no layers)
                                   # TODO: Handle turn counts for screens/tailwind if needed
                     else: print(f"Warning: Could not parse player/condition in |-sidestart| line: {line.strip()}")
                 else: print(f"Warning: Malformed |-sidestart| line: {line.strip()}")


            elif event_type == '-sideend':
                # Example: |-sideend|p1: PlayerName|move: Reflect|
                if len(parts) >= 4:
                    side_identifier, condition_str = parts[2], parts[3]
                    affected_side_player = parse_player_from_side(side_identifier)
                    condition = parse_condition_name(condition_str)

                    if affected_side_player and condition:
                        target_side_key = f'{affected_side_player}_side'
                        # Only handles non-hazard side conditions ending
                        if condition in SIDE_CONDITIONS and condition not in HAZARD_CONDITIONS and target_side_key in current_state['field']:
                             side_dict = current_state['field'][target_side_key]
                             if condition in side_dict:
                                  side_dict[condition] = 0 # Mark as ended
                        # Note: Hazards are typically removed by moves like Defog/Rapid Spin, not |-sideend|
                    else: print(f"Warning: Could not parse player/condition in |-sideend| line: {line.strip()}")
                else: print(f"Warning: Malformed |-sideend| line: {line.strip()}")

            # --- Action Recording ---
            is_player_action = False
            action_description = None
            acting_player = None
            move_used_this_action = None # Store move name if this action is a move

            if event_type == 'move':
                # Example: |move|p1a: Zapdos|Thunderbolt|p2a: Slowking|[from]lockedmove
                if len(parts) >= 4:
                    identifier, move_name_raw, *_ = parts[2:]
                    # Clean move name (remove [from], [still], etc.)
                    move_name = re.sub(r'\[from\].*|\[still\]', '', move_name_raw).strip()
                    player, ident_name_part = parse_pokemon_identifier(identifier)
                    if player:
                        acting_player = player
                        is_player_action = True
                        action_description = f"move:{move_name}"
                        move_used_this_action = move_name # Record the move name

                        # --- Update nickname map if move identifier reveals it ---
                        slot_that_moved = active_slot.get(acting_player)
                        if slot_that_moved and slot_that_moved in slot_id_to_species.get(acting_player, {}):
                            initial_species = slot_id_to_species[acting_player][slot_that_moved]
                            # If identifier name is different and not already mapped, map it
                            if ident_name_part and ident_name_part != initial_species:
                                 # Only map if not already present or if mapping to the same slot
                                 current_mapping = nickname_to_slot_id.get(acting_player, {}).get(ident_name_part)
                                 if current_mapping is None or current_mapping == slot_that_moved:
                                      nickname_to_slot_id.setdefault(acting_player, {})[ident_name_part] = slot_that_moved
                                      # print(f"DEBUG: Mapped nickname '{ident_name_part}' from move event to slot '{slot_that_moved}' for player {acting_player}") # Optional Debug
                else: print(f"Warning: Malformed |move| line: {line.strip()}")


            elif event_type == 'switch': # Switch is also a player action
                # Example: |switch|p1a: Landorus-Therian|Landorus-Therian, L85, M|100/100
                if len(parts) >= 4: # Need at least |switch|pXa: Pokemon|details|hp
                    identifier, details, *_ = parts[2:]
                    player, ident_name_part = parse_pokemon_identifier(identifier)
                    # Get the species being switched TO from the details part
                    species = details.split(',')[0].replace('-*', '')
                    if player:
                        acting_player = player
                        is_player_action = True
                        action_description = f"switch:{species}"
                        # Nickname mapping is handled within the main switch processing block above
                else: print(f"Warning: Malformed |switch| line (for action recording): {line.strip()}")


            # --- State Snapshotting and Update Last Move / Revealed Moves ---
            if is_player_action and acting_player:
                # --- Create snapshot BEFORE updating revealed_moves or last_move_seen ---
                state_snapshot = copy.deepcopy(current_state)

                # Store the *previously* seen last moves in the snapshot
                state_snapshot['last_move_p1'] = last_move_seen['p1']
                state_snapshot['last_move_p2'] = last_move_seen['p2']

                # Store the action taken and the player who took it in the snapshot
                state_snapshot['action_taken'] = action_description
                state_snapshot['player_to_move'] = acting_player # The player who committed to this action

                # Add the snapshot to our list of game states
                game_states.append(state_snapshot)

                # --- Update main state AFTER snapshotting ---

                # 1. Update last_move_seen tracker for the player who acted
                if move_used_this_action:
                    last_move_seen[acting_player] = move_used_this_action
                # Switches don't update the 'last_move_seen' directly here,
                # as it tracks the last *move command*.

                # 2. Update revealed_moves for the Pokémon that used a move
                if move_used_this_action:
                    slot_that_moved = active_slot.get(acting_player)
                    if slot_that_moved and acting_player in current_state and slot_that_moved in current_state[acting_player]:
                        poke_state = current_state[acting_player].get(slot_that_moved)
                        # Ensure poke_state and revealed_moves set exist
                        if poke_state and isinstance(poke_state.get('revealed_moves'), set):
                             poke_state['revealed_moves'].add(move_used_this_action)
                        # else: # Debugging if needed
                        #      print(f"Warning: Could not find poke_state or revealed_moves set for {acting_player}/{slot_that_moved} when adding move '{move_used_this_action}'.")


                # Update player expected to move next (simplistic alternation placeholder)
                # The 'player_to_move' in the snapshot is the accurate one for the decision point.
                current_state['player_to_move'] = 'p2' if acting_player == 'p1' else 'p1'
                last_action_player = acting_player # Track who just acted

        except Exception as e:
             print(f"\n!!! PARSING ERROR on line {line_num+1} for Replay ID {replay_id} !!!")
             print(f"Line content: {line.strip()}")
             print(f"Error type: {type(e).__name__}")
             print(f"Error message: {e}")
             import traceback
             print("Traceback:")
             traceback.print_exc()
             # Decide whether to continue parsing the rest of the file or stop
             # For robustness, let's try to continue
             print("--- Continuing parsing ---")
             # You might want to mark this replay as problematic


    # --- Finalize --- (Add winner to all states, ensure defaults)
    final_winner = current_state.get('battle_winner')
    for state in game_states:
        state['battle_winner'] = final_winner
        # Ensure last moves fields exist even if no moves happened before first action
        state.setdefault('last_move_p1', 'none')
        state.setdefault('last_move_p2', 'none')

    return game_states


# --- Flattening function ---
def flatten_state(state_dict):
    """Converts a nested state dictionary into a flat dictionary for DataFrame creation."""
    flat = {}
    flat['replay_id'] = state_dict.get('replay_id', 'unknown')
    flat['turn_number'] = state_dict.get('turn_number', 0)
    flat['player_to_move'] = state_dict.get('player_to_move') # Player who took the action AFTER this state
    flat['action_taken'] = state_dict.get('action_taken')     # Action taken AFTER this state (target variable)
    flat['battle_winner'] = state_dict.get('battle_winner')
    flat['last_move_p1'] = state_dict.get('last_move_p1', 'none') # Move used by P1 before this state
    flat['last_move_p2'] = state_dict.get('last_move_p2', 'none') # Move used by P2 before this state

    # Pokemon States
    for player in ['p1', 'p2']:
        # --- ALWAYS LOOP 6 TIMES for standard singles ---
        num_slots_to_generate = 6 # Assume standard 6 slots for output structure
        for i in range(num_slots_to_generate):
             slot_id = f'slot{i+1}'
             prefix = f'{player}_{slot_id}'
             # Check if the player and slot exist in the actual state dictionary
             if player in state_dict and isinstance(state_dict[player], dict) and slot_id in state_dict[player]:
                 poke_state = state_dict[player][slot_id]
                 # Check if poke_state is a valid dictionary
                 if isinstance(poke_state, dict):
                     flat[f'{prefix}_species'] = poke_state.get('species', 'Unknown')
                     flat[f'{prefix}_hp_perc'] = poke_state.get('hp_perc', 100) # Default HP
                     flat[f'{prefix}_status'] = poke_state.get('status', 'none')
                     flat[f'{prefix}_is_active'] = int(poke_state.get('is_active', False))
                     flat[f'{prefix}_is_fainted'] = int(poke_state.get('is_fainted', False))
                     flat[f'{prefix}_terastallized'] = int(poke_state.get('terastallized', False))
                     flat[f'{prefix}_tera_type'] = poke_state.get('tera_type', 'none')
                     # Boosts
                     boosts = poke_state.get('boosts', {})
                     for stat in ['atk', 'def', 'spa', 'spd', 'spe']:
                         flat[f'{prefix}_boost_{stat}'] = boosts.get(stat, 0)
                     # Revealed Moves
                     moves_set = poke_state.get('revealed_moves', set())
                     # Ensure it's actually a set before processing
                     if isinstance(moves_set, set):
                          # Sort for consistency and join into a string (or use 'none')
                          flat[f'{prefix}_revealed_moves'] = ",".join(sorted(list(moves_set))) if moves_set else 'none'
                     else:
                          print(f"Warning: '{prefix}_revealed_moves' was not a set ({type(moves_set)}), defaulting to 'error_state'. Replay: {flat['replay_id']}")
                          flat[f'{prefix}_revealed_moves'] = 'error_state' # Indicate bad data
                 else:
                      # Log warning if poke_state is not a dict (error condition)
                      print(f"Warning: Expected dict for poke_state {player}/{slot_id}, got {type(poke_state)}. Replay: {flat['replay_id']}")
                      # Fill with default/error values
                      flat[f'{prefix}_species'] = 'ErrorState'
                      flat[f'{prefix}_hp_perc'] = 0
                      flat[f'{prefix}_status'] = 'none'
                      flat[f'{prefix}_is_active'] = 0
                      flat[f'{prefix}_is_fainted'] = 1
                      flat[f'{prefix}_terastallized'] = 0
                      flat[f'{prefix}_tera_type'] = 'none'
                      for stat in ['atk', 'def', 'spa', 'spd', 'spe']: flat[f'{prefix}_boost_{stat}'] = 0
                      flat[f'{prefix}_revealed_moves'] = 'error_state'
             else:
                  # Add placeholder columns if the slot is missing from the state dict
                  # (handles smaller team sizes correctly now)
                  flat[f'{prefix}_species'] = 'Absent' # Indicate slot doesn't exist or wasn't parsed
                  flat[f'{prefix}_hp_perc'] = 0
                  flat[f'{prefix}_status'] = 'none'
                  flat[f'{prefix}_is_active'] = 0
                  flat[f'{prefix}_is_fainted'] = 1 # Assume absent means unavailable/fainted
                  flat[f'{prefix}_terastallized'] = 0
                  flat[f'{prefix}_tera_type'] = 'none'
                  for stat in ['atk', 'def', 'spa', 'spd', 'spe']: flat[f'{prefix}_boost_{stat}'] = 0
                  flat[f'{prefix}_revealed_moves'] = 'none' # No moves revealed if absent

    # Field States (This part was likely okay, but ensure consistency)
    initial_field = get_initial_field_state() # Use for defaults and checking keys
    if 'field' in state_dict and isinstance(state_dict['field'], dict):
        field_state = state_dict['field']
        flat['field_weather'] = field_state.get('weather', initial_field['weather'])
        flat['field_terrain'] = field_state.get('terrain', initial_field['terrain'])
        flat['field_pseudo_weather'] = field_state.get('pseudo_weather', initial_field['pseudo_weather'])

        # Hazards - Ensure all expected hazard columns are created
        for player_prefix, hazard_key in [('p1', 'p1_hazards'), ('p2', 'p2_hazards')]:
            hazard_dict = field_state.get(hazard_key, initial_field[hazard_key]) # Use initial as default
            for hazard in HAZARD_CONDITIONS: # Iterate through known hazards
                 flat_key = f'{player_prefix}_hazard_{hazard.lower().replace(" ", "_")}'
                 flat[flat_key] = hazard_dict.get(hazard, 0) # Get value or default to 0

        # Side Conditions - Ensure all expected side condition columns are created
        relevant_side_conditions = SIDE_CONDITIONS - HAZARD_CONDITIONS
        for player_prefix, side_key in [('p1', 'p1_side'), ('p2', 'p2_side')]:
            side_dict = field_state.get(side_key, initial_field[side_key]) # Use initial as default
            for condition in relevant_side_conditions: # Iterate through known side conditions
                 flat_key = f'{player_prefix}_side_{condition.lower().replace(" ", "_")}'
                 flat[flat_key] = side_dict.get(condition, 0) # Get value or default to 0
    else:
         # Add default field states if 'field' key is missing or invalid
         if 'field' not in state_dict:
              print(f"Warning: Missing 'field' key in state for replay {flat['replay_id']}, turn {flat['turn_number']}. Filling defaults.")
         else:
              print(f"Warning: Invalid 'field' value (type {type(state_dict['field'])}) in state for replay {flat['replay_id']}, turn {flat['turn_number']}. Filling defaults.")

         flat['field_weather'] = initial_field['weather']
         flat['field_terrain'] = initial_field['terrain']
         flat['field_pseudo_weather'] = initial_field['pseudo_weather']
         # Default hazards
         for player_prefix, hazard_key in [('p1', 'p1_hazards'), ('p2', 'p2_hazards')]:
              for hazard in HAZARD_CONDITIONS: flat[f'{player_prefix}_hazard_{hazard.lower().replace(" ", "_")}'] = 0
         # Default side conditions
         relevant_side_conditions = SIDE_CONDITIONS - HAZARD_CONDITIONS
         for player_prefix, side_key in [('p1', 'p1_side'), ('p2', 'p2_side')]:
              for condition in relevant_side_conditions: flat[f'{player_prefix}_side_{condition.lower().replace(" ", "_")}'] = 0

    return flat


# --- Main Script Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse Pokemon Showdown replay log files into a single data table (Parquet/CSV).")
    parser.add_argument("input_dir", type=str, help="Directory containing the downloaded .log files.")
    parser.add_argument("output_file", type=str, help="Path to save the output data table (e.g., output.parquet or output.csv).")
    parser.add_argument("--format", choices=['parquet', 'csv'], default='parquet', help="Output file format (parquet or csv). Parquet recommended.")
    parser.add_argument("--limit", type=int, default=None, help="Optional: Limit the number of files to process (for testing).")

    args = parser.parse_args()

    log_files = glob.glob(os.path.join(args.input_dir, '*.log'))
    if not log_files:
        print(f"Error: No .log files found in directory: {args.input_dir}")
        exit(1)

    # Apply limit if specified
    if args.limit:
        print(f"Limiting processing to the first {args.limit} files found.")
        log_files = log_files[:args.limit]

    print(f"Found {len(log_files)} .log files to parse in '{args.input_dir}'.")

    all_flattened_states = []
    parsed_files = 0
    failed_files = 0

    # Loop through log files, parse, flatten, and collect states
    for i, file_path in enumerate(log_files):
        filename = os.path.basename(file_path)
        # Attempt to extract numeric replay ID from filename for better tracking
        match = re.search(r'-(\d+)\.log$', filename)
        replay_id = match.group(1) if match else f"unknown_{filename}"

        print(f"Parsing file {i+1}/{len(log_files)}: {filename} (ID: {replay_id}) ...", end=' ', flush=True)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                log_content = f.read()

            # Parse the replay log
            parsed_replay_states = parse_showdown_replay(log_content, replay_id=replay_id)

            if not parsed_replay_states:
                print("Warning: No action states extracted.")
                # Decide if this counts as a failure or just an empty/unparseable replay
                # failed_files += 1 # Option 1: Count as failure
                continue        # Option 2: Skip without counting as failure

            # Flatten each state dictionary
            flattened_replay_states = [flatten_state(s) for s in parsed_replay_states]

            # Add the flattened states to the main list
            all_flattened_states.extend(flattened_replay_states)
            print(f"OK ({len(flattened_replay_states)} states added).")
            parsed_files += 1

        except FileNotFoundError:
            print(f"Error: File not found: {filename}")
            failed_files += 1
        except Exception as e:
            # Catch any unexpected errors during parsing or flattening
            print(f"FAIL. Unexpected Error: {e}")
            failed_files += 1
            # Optionally print detailed traceback for debugging
            import traceback
            traceback.print_exc()

    # --- Post-Parsing Summary ---
    print(f"\nParsing complete.")
    print(f"Successfully parsed (produced states): {parsed_files} files")
    print(f"Failed or produced no states:        {failed_files + (len(log_files) - parsed_files - failed_files)} files")
    print(f"Total states collected:              {len(all_flattened_states)}")

    if not all_flattened_states:
        print("No states collected, skipping output.")
        exit(0)

    # --- DataFrame Creation and Saving ---
    print("\nCreating DataFrame...")
    try:
        df = pd.DataFrame(all_flattened_states)
        print("DataFrame created successfully.")

        # --- Optional: Data Cleaning/Validation ---
        # Example: Fill NaN values based on expected dtype
        # print("Attempting to fill NaN values...")
        # for col in df.columns:
        #     if pd.api.types.is_numeric_dtype(df[col]):
        #         df[col].fillna(0, inplace=True) # Fill numeric NaNs with 0
        #     elif pd.api.types.is_object_dtype(df[col]):
        #          # Fill object NaNs with 'none' or specific placeholder
        #         df[col].fillna('none', inplace=True)
        # print("NaN filling complete (if any were present).")

        # --- Column Checks ---
        print("\nChecking for key columns in DataFrame:")
        print(f"  'last_move_p1' exists: {'last_move_p1' in df.columns}")
        print(f"  'last_move_p2' exists: {'last_move_p2' in df.columns}")
        revealed_cols_exist = any(col.endswith('_revealed_moves') for col in df.columns)
        print(f"  '*_revealed_moves' columns exist: {revealed_cols_exist}")
        if revealed_cols_exist:
             example_col = next((col for col in df.columns if col.endswith('_revealed_moves')), None)
             if example_col:
                  print(f"\n  Sample values for a revealed moves column ('{example_col}'):")
                  value_counts = df[example_col].value_counts()
                  # Show non-'none' values preferentially
                  non_none_values = value_counts[value_counts.index != 'none']
                  if not non_none_values.empty:
                     print(non_none_values.head())
                     print(f"    (Showing top non-'none' values)")
                  # Also show count of 'none'
                  none_count = value_counts.get('none', 0)
                  error_count = value_counts.get('error_state', 0)
                  print(f"    Count of 'none': {none_count}")
                  if error_count > 0: print(f"    Count of 'error_state': {error_count}")


        print(f"\nSaving DataFrame to '{args.output_file}' (format: {args.format})...")
        if args.format == 'parquet':
            # Use pyarrow engine, consider schema consistency options if needed
            df.to_parquet(args.output_file, index=False, engine='pyarrow')
        elif args.format == 'csv':
            df.to_csv(args.output_file, index=False)

        print("DataFrame saved successfully.")
        print("\nFinal DataFrame Info:")
        # Show more columns in df.info()
        pd.set_option('display.max_info_columns', 200)
        pd.set_option('display.max_rows', 200) # Also potentially useful
        df.info(verbose=True, show_counts=True)
        # Display sample data
        print("\nDataFrame Head:")
        print(df.head())
        print("\nDataFrame Tail:")
        print(df.tail())
        # Reset display options
        pd.reset_option('display.max_info_columns')
        pd.reset_option('display.max_rows')


    except Exception as e:
        print(f"\nError creating or saving DataFrame: {e}")
        import traceback
        traceback.print_exc()
        # Attempt emergency CSV save if Parquet failed
        if args.format == 'parquet' and 'df' in locals(): # Check if df exists
             try:
                  alt_path = args.output_file.replace('.parquet', '.emergency.csv')
                  print(f"\nAttempting to save as emergency CSV to: {alt_path}")
                  df.to_csv(alt_path, index=False)
                  print("Emergency CSV saved.")
             except Exception as csv_e:
                  print(f"Failed to save emergency CSV: {csv_e}")
        exit(1)

# --- END OF (COMPLETE Modified) process_replays.py ---