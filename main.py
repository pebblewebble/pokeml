import pandas as pd
import re
import copy
import uuid # For generating unique replay IDs if not present

def parse_hp(hp_str):
    """Parses HP string like '100/100' or 'fnt' into percentage."""
    if hp_str == 'fnt':
        return 0
    try:
        parts = hp_str.split('/')
        if len(parts) == 2:
            return round((int(parts[0]) / int(parts[1])) * 100) if int(parts[1]) > 0 else 0
        # Handle percentage format directly if present (e.g., from |-damage|)
        return int(hp_str)
    except (ValueError, TypeError, IndexError):
        return None # Or some indicator of unknown HP

def get_initial_pokemon_state():
    """Returns a default structure for a single Pokémon's state."""
    return {
        'species': 'Unknown',
        'hp_perc': 100,
        'status': 'none',
        'is_active': False,
        'is_fainted': False,
        'boosts': {'atk': 0, 'def': 0, 'spa': 0, 'spd': 0, 'spe': 0},
        'terastallized': False,
        'tera_type': 'none',
        'revealed': False # Track if we've seen this Pokemon yet
    }

def get_initial_field_state():
    """Returns default field state."""
    return {
        'weather': 'none',
        'weather_turns': 0, # Simplified tracking
        'terrain': 'none',
        'terrain_turns': 0, # Simplified tracking
        'p1_hazards': {'Stealth Rock': 0, 'Spikes': 0, 'Toxic Spikes': 0, 'Sticky Web': 0},
        'p2_hazards': {'Stealth Rock': 0, 'Spikes': 0, 'Toxic Spikes': 0, 'Sticky Web': 0},
        # Add screens if needed: 'p1_screens': {'Reflect': 0, 'Light Screen': 0}, etc.
    }

def parse_pokemon_identifier(identifier):
    """Parses 'p1a: Ninetales' into ('p1', 'Ninetales')."""
    match = re.match(r'(p[12])[a-z]?: (.*)', identifier)
    if match:
        player, nickname = match.groups()
        # Basic handling if species name is in nickname (less reliable)
        # A better parser would use the initial |poke| info mapped to nicknames
        return player, nickname.strip()
    return None, None

# --- Main Parsing Function ---
def parse_showdown_replay(log_text, replay_id=None):
    """
    Parses a Pokémon Showdown replay log into a list of state-action dictionaries.

    Args:
        log_text (str): The full replay log as a single string.
        replay_id (str, optional): A unique ID for this replay. Auto-generates if None.

    Returns:
        list: A list of dictionaries, where each dictionary represents a game state
              before a player action.
        dict: Final state at the end of the battle (for debugging or outcome).
    """
    if replay_id is None:
        replay_id = str(uuid.uuid4())

    lines = log_text.strip().split('\n')
    
    game_states = []
    current_state = {
        'replay_id': replay_id,
        'turn_number': 0,
        'player_to_move': None, # Will track who is expected to act
        'p1': {}, # Holds state for p1's canonical slots
        'p2': {}, # Holds state for p2's canonical slots
        'field': get_initial_field_state(),
        'action_taken': None,
        'battle_winner': None
    }
    
    # --- Phase 1: Initial Setup (Players, Teams) ---
    player_names = {'p1': None, 'p2': None}
    teams = {'p1': {}, 'p2': {}} # Maps slot_id -> species
    nickname_to_player_slot = {} # Maps nickname -> (player, slot_id) for quick lookup
    active_pokemon_nicknames = {'p1': None, 'p2': None} # Tracks current nickname on field

    current_teamsize = {'p1': 0, 'p2': 0}
    slot_counters = {'p1': 0, 'p2': 0}

    for line in lines:
        parts = line.strip().split('|')
        if len(parts) < 2: continue
        
        event_type = parts[1]

        if event_type == 'player':
            player_id, name, *_ = parts[2:]
            player_names[player_id] = name
        elif event_type == 'teamsize':
            player_id, size = parts[2], int(parts[3])
            current_teamsize[player_id] = size
            # Initialize player slots in current_state
            current_state[player_id] = {f'slot{i+1}': get_initial_pokemon_state() for i in range(size)}
        elif event_type == 'poke':
            player_id, details, *_ = parts[2:]
            species = details.split(',')[0].replace('-*', '') # Handle Zama-* etc.
            slot_counters[player_id] += 1
            slot_id = f'slot{slot_counters[player_id]}'
            if slot_counters[player_id] <= current_teamsize.get(player_id, 6): # Ensure within bounds
                teams[player_id][slot_id] = species
                current_state[player_id][slot_id]['species'] = species

    # Helper to find the slot_id for an active pokemon nickname
    def find_slot_id(player, nickname):
         # This mapping needs to be built more reliably during switches/drags
         # For now, we find the first slot matching the species of the nickname
         # (This is brittle if multiple of same species exist and nicknames differ)
        clean_nickname_species = nickname.split(',')[0].replace('-*', '') # Crude way to get species from nickname field
        for slot, data in current_state[player].items():
             # Check if species matches AND if it could be the active one (not fainted, maybe check revealed)
             # A better parser links nickname from |switch| to slot definitively
            if data['species'] == clean_nickname_species and not data['is_fainted']:
                # A temporary fix: assume the first non-fainted match is correct
                # This NEEDS improvement for robustness
                return slot
        # Fallback: if nickname is the species name itself
        for slot, data in current_state[player].items():
            if data['species'] == nickname and not data['is_fainted']:
                 return slot
        print(f"Warning: Could not reliably map nickname '{nickname}' for {player} to a slot.")
        return None # Could not find mapping


    # --- Phase 2: Process Turns and Events ---
    last_action_player = None # Track who acted last to infer next player_to_move

    for line in lines:
        parts = line.strip().split('|')
        if len(parts) < 2: continue

        event_type = parts[1]
        
        # --- State Updates ---
        if event_type == 'turn':
            current_state['turn_number'] = int(parts[2])
            # Reset who moved this turn, typically p1 starts turn 1 unless lead shenanigans
            last_action_player = 'p2' if current_state['turn_number'] > 0 else None
            current_state['player_to_move'] = 'p1' # Assume p1 starts (simplification)

        elif event_type in ['switch', 'drag']:
            identifier, details, hp_str = parts[2], parts[3], parts[4]
            player, nickname = parse_pokemon_identifier(identifier)
            species = details.split(',')[0].replace('-*', '') # Get species
            
            # Deactivate previous pokemon for this player
            if active_pokemon_nicknames[player]:
                 old_active_slot_id = find_slot_id(player, active_pokemon_nicknames[player])
                 if old_active_slot_id:
                     current_state[player][old_active_slot_id]['is_active'] = False

            # Find slot and update new active pokemon
            slot_id = find_slot_id(player, nickname) # Attempt to find based on species
            if not slot_id: # If couldn't find, try adding based on species if missing
                 print(f"Debug: Trying species match for {nickname} ({species})")
                 found = False
                 for s_id, data in current_state[player].items():
                     if data['species'] == species and not data['is_fainted']:
                         slot_id = s_id
                         found = True
                         break
                 if not found:
                    print(f"ERROR: Could not find slot for {player}'s {nickname} ({species}) during {event_type}")
                    continue


            hp_perc = parse_hp(hp_str)
            current_state[player][slot_id]['species'] = species # Ensure species is correct
            current_state[player][slot_id]['hp_perc'] = hp_perc
            current_state[player][slot_id]['is_active'] = True
            current_state[player][slot_id]['is_fainted'] = (hp_perc == 0)
            current_state[player][slot_id]['revealed'] = True
            # Reset boosts on switch (simplification, ignores Baton Pass)
            current_state[player][slot_id]['boosts'] = {k: 0 for k in current_state[player][slot_id]['boosts']}
            current_state[player][slot_id]['terastallized'] = False # Tera resets on switch
            current_state[player][slot_id]['tera_type'] = 'none'

            active_pokemon_nicknames[player] = nickname
            nickname_to_player_slot[nickname] = (player, slot_id) # Update mapping

        elif event_type == '-damage' or event_type == '-heal':
            identifier, hp_str = parts[2], parts[3]
            player, nickname = parse_pokemon_identifier(identifier)
            slot_id = find_slot_id(player, nickname)
            if slot_id:
                hp_perc = parse_hp(hp_str)
                current_state[player][slot_id]['hp_perc'] = hp_perc
                if hp_perc == 0:
                    current_state[player][slot_id]['status'] = 'fnt'
                    current_state[player][slot_id]['is_fainted'] = True
                    current_state[player][slot_id]['is_active'] = False # Fainted pokemon is not active
            else:
                 print(f"Warning: Couldn't find slot for {nickname} during {event_type}")


        elif event_type == '-status':
            identifier, status = parts[2], parts[3]
            player, nickname = parse_pokemon_identifier(identifier)
            slot_id = find_slot_id(player, nickname)
            if slot_id:
                current_state[player][slot_id]['status'] = status
            else:
                 print(f"Warning: Couldn't find slot for {nickname} during {event_type}")


        elif event_type == '-curestatus':
            identifier, status = parts[2], parts[3]
            player, nickname = parse_pokemon_identifier(identifier)
            slot_id = find_slot_id(player, nickname)
            if slot_id and current_state[player][slot_id]['status'] == status:
                current_state[player][slot_id]['status'] = 'none'
            elif slot_id:
                # Could be curing confusion, which isn't tracked here, ignore for now
                pass
            else:
                 print(f"Warning: Couldn't find slot for {nickname} during {event_type}")


        elif event_type == '-boost' or event_type == '-unboost':
            identifier, stat, amount = parts[2], parts[3], int(parts[4])
            player, nickname = parse_pokemon_identifier(identifier)
            slot_id = find_slot_id(player, nickname)
            if slot_id and stat in current_state[player][slot_id]['boosts']:
                multiplier = 1 if event_type == '-boost' else -1
                current_state[player][slot_id]['boosts'][stat] += (multiplier * amount)
                # Clamp boosts between -6 and +6
                current_state[player][slot_id]['boosts'][stat] = max(-6, min(6, current_state[player][slot_id]['boosts'][stat]))
            elif slot_id:
                 # Stat not tracked (e.g., accuracy, evasion)
                 pass
            else:
                 print(f"Warning: Couldn't find slot for {nickname} during {event_type}")

        elif event_type == '-weather':
            weather_type = parts[2]
            if weather_type == 'none':
                 current_state['field']['weather'] = 'none'
                 current_state['field']['weather_turns'] = 0
            else:
                 # Extract weather type if details are present (e.g., "[from] ability: Drought")
                 current_state['field']['weather'] = weather_type.split(' ')[0]
                 # Simplified turn count - needs proper tracking based on source
                 current_state['field']['weather_turns'] = 5 # Default, adjust for Drought etc. if needed


        elif event_type == '-sidestart':
             side, condition = parts[2], parts[3] # e.g., 'p1', 'Stealth Rock'
             player = side # Should be 'p1' or 'p2'
             condition_name = condition.split(': ')[-1] # Get 'Stealth Rock' from 'move: Stealth Rock'

             hazard_map = { # Map log condition names to our state keys
                'Stealth Rock': 'Stealth Rock',
                'Spikes': 'Spikes',
                'Toxic Spikes': 'Toxic Spikes',
                'Sticky Web': 'Sticky Web'
             }
             target_player_hazards = 'p1_hazards' if player == 'p2' else 'p2_hazards' # Hazards affect opponent

             if condition_name in hazard_map:
                 state_key = hazard_map[condition_name]
                 if state_key in ['Spikes', 'Toxic Spikes']:
                      current_state['field'][target_player_hazards][state_key] = min(3 if state_key == 'Spikes' else 2,
                                                                                    current_state['field'][target_player_hazards][state_key] + 1)
                 else: # SR, Web are binary (or count=1)
                      current_state['field'][target_player_hazards][state_key] = 1


        elif event_type == 'faint':
            identifier = parts[2]
            player, nickname = parse_pokemon_identifier(identifier)
            slot_id = find_slot_id(player, nickname)
            if slot_id:
                current_state[player][slot_id]['hp_perc'] = 0
                current_state[player][slot_id]['status'] = 'fnt'
                current_state[player][slot_id]['is_fainted'] = True
                current_state[player][slot_id]['is_active'] = False
            else:
                 print(f"Warning: Couldn't find slot for fainted {nickname}")

        elif event_type == '-terastallize':
             identifier, tera_type = parts[2], parts[3]
             player, nickname = parse_pokemon_identifier(identifier)
             slot_id = find_slot_id(player, nickname)
             if slot_id:
                  current_state[player][slot_id]['terastallized'] = True
                  current_state[player][slot_id]['tera_type'] = tera_type
             else:
                  print(f"Warning: Couldn't find slot for {nickname} during {event_type}")

        elif event_type == 'win':
            winner_name = parts[2]
            if player_names['p1'] == winner_name:
                 current_state['battle_winner'] = 'p1'
            elif player_names['p2'] == winner_name:
                 current_state['battle_winner'] = 'p2'


        # --- Action Recording ---
        # Record state *before* the action is fully processed
        # This logic determines *when* to save a snapshot
        is_player_action = False
        action_description = None

        if event_type == 'move':
            identifier, move_name, *_ = parts[2:]
            player, nickname = parse_pokemon_identifier(identifier)
            if player:
                is_player_action = True
                action_description = f"move:{move_name}"
                current_state['player_to_move'] = player # Set who just moved

        elif event_type == 'switch': # This is also a player action
            identifier, details, *_ = parts[2:]
            player, nickname = parse_pokemon_identifier(identifier)
            species = details.split(',')[0].replace('-*', '')
            if player:
                 is_player_action = True
                 action_description = f"switch:{species}"
                 current_state['player_to_move'] = player # Set who just moved


        if is_player_action:
            # Determine whose turn it *was* based on who just acted
            acting_player = current_state['player_to_move']

            # Create a deep copy of the state *before* this action
            state_snapshot = copy.deepcopy(current_state)
            state_snapshot['action_taken'] = action_description
            # Set player_to_move in the snapshot to who was *about* to move
            state_snapshot['player_to_move'] = acting_player

            game_states.append(state_snapshot)

            # Infer next player to move (simple alternating logic)
            last_action_player = acting_player
            current_state['player_to_move'] = 'p2' if acting_player == 'p1' else 'p1'


    # --- Phase 3: Finalize ---
    # Add winner info to all recorded states retroactively
    final_winner = current_state.get('battle_winner')
    for state in game_states:
        state['battle_winner'] = final_winner

    return game_states, current_state # Return list of state dicts and the final state


def flatten_state(state_dict):
    """Flattens the nested state dictionary into a single-level dictionary."""
    flat = {}
    flat['replay_id'] = state_dict['replay_id']
    flat['turn_number'] = state_dict['turn_number']
    flat['player_to_move'] = state_dict['player_to_move'] # Who was about to move
    flat['action_taken'] = state_dict['action_taken']
    flat['battle_winner'] = state_dict['battle_winner']

    for player in ['p1', 'p2']:
        if player in state_dict:
             num_slots = len(state_dict[player])
             for i in range(num_slots):
                 slot_id = f'slot{i+1}'
                 if slot_id in state_dict[player]:
                     poke_state = state_dict[player][slot_id]
                     flat[f'{player}_{slot_id}_species'] = poke_state['species']
                     flat[f'{player}_{slot_id}_hp_perc'] = poke_state['hp_perc']
                     flat[f'{player}_{slot_id}_status'] = poke_state['status']
                     flat[f'{player}_{slot_id}_is_active'] = int(poke_state['is_active']) # Use 1/0
                     flat[f'{player}_{slot_id}_is_fainted'] = int(poke_state['is_fainted'])# Use 1/0
                     flat[f'{player}_{slot_id}_terastallized'] = int(poke_state['terastallized'])
                     flat[f'{player}_{slot_id}_tera_type'] = poke_state['tera_type']
                     for stat, value in poke_state['boosts'].items():
                         flat[f'{player}_{slot_id}_boost_{stat}'] = value
                 else:
                     # Handle cases where slot might be missing if teamsize parsing failed
                     pass

    if 'field' in state_dict:
        flat['field_weather'] = state_dict['field']['weather']
        # flat['field_weather_turns'] = state_dict['field']['weather_turns'] # Optional
        flat['field_terrain'] = state_dict['field']['terrain']
        # flat['field_terrain_turns'] = state_dict['field']['terrain_turns'] # Optional
        for side in ['p1', 'p2']:
             hazard_dict = state_dict['field'].get(f'{side}_hazards', {})
             for hazard, value in hazard_dict.items():
                  flat[f'{side}_hazard_{hazard.lower().replace(" ", "_")}'] = value

    return flat


# --- Example Usage ---
replay_log = """
|badge|p1|gold|gen9ou|3-2
|uhtml|medal-msg|<div class="broadcast-blue">Curious what those medals under the avatar are? PS now has Ladder Seasons! For more information, check out the <a href="https://www.smogon.com/forums/threads/3740067/">thread on Smogon.</a></div>
|badge|p2|gold|gen9ou|3-2
|uhtml|medal-msg|<div class="broadcast-blue">Curious what those medals under the avatar are? PS now has Ladder Seasons! For more information, check out the <a href="https://www.smogon.com/forums/threads/3740067/">thread on Smogon.</a></div>
|raw|<div class="broadcast-blue"><strong>This battle is required to be public due to a player having a name starting with 'lt111vz'.</div>
|j|☆GALAXY☆00000
|j|☆LT111VZ bewashed
|t:|1722616707
|gametype|singles
|player|p1|GALAXY☆00000|lucas|2128
|player|p2|LT111VZ bewashed|youngcouple-gen4dp|2129
|teamsize|p1|6
|teamsize|p2|6
|gen|9
|tier|[Gen 9] OU
|rated|
|rule|Species Clause: Limit one of each Pokémon
|rule|OHKO Clause: OHKO moves are banned
|rule|Evasion Items Clause: Evasion items are banned
|rule|Evasion Moves Clause: Evasion moves are banned
|rule|Endless Battle Clause: Forcing endless battles is banned
|rule|HP Percentage Mod: HP is shown in percentages
|rule|Sleep Moves Clause: Sleep-inducing moves are banned
|clearpoke
|poke|p1|Zamazenta-*|
|poke|p1|Sandy Shocks|
|poke|p1|Gouging Fire|
|poke|p1|Ninetales, F|
|poke|p1|Landorus-Therian, M|
|poke|p1|Walking Wake|
|poke|p2|Kyurem|
|poke|p2|Cinderace, M|
|poke|p2|Zapdos|
|poke|p2|Tinkaton, F|
|poke|p2|Great Tusk|
|poke|p2|Roaring Moon|
|teampreview
|inactive|Battle timer is ON: inactive players will automatically lose when time's up. (requested by GALAXY☆00000)
|j| xuanse123
|c|☆LT111VZ bewashed|CMON WHY VERT
|c|☆GALAXY☆00000|hf
|c|☆LT111VZ bewashed|Ups
|c|☆LT111VZ bewashed|sorry
|j| Sujay532
|inactive|LT111VZ bewashed has 120 seconds left.
|j| jeieooe9w
|
|t:|1722616744
|start
|switch|p1a: Ninetales|Ninetales, F|100/100
|switch|p2a: Kyurem|Kyurem|100/100
|-weather|SunnyDay|[from] ability: Drought|[of] p1a: Ninetales
|-ability|p2a: Kyurem|Pressure
|turn|1
|j| pokhikjl
|
|t:|1722616751
|move|p1a: Ninetales|Weather Ball|p2a: Kyurem
|-damage|p2a: Kyurem|51/100
|move|p2a: Kyurem|Substitute|p2a: Kyurem
|-start|p2a: Kyurem|Substitute
|-damage|p2a: Kyurem|26/100
|
|-weather|SunnyDay|[upkeep]
|-heal|p2a: Kyurem|32/100|[from] item: Leftovers
|upkeep
|turn|2
|j|+thankyou
|j| lt111vz smoolic
|
|t:|1722616755
|move|p2a: Kyurem|Protect|p2a: Kyurem
|-singleturn|p2a: Kyurem|Protect
|move|p1a: Ninetales|Encore|p2a: Kyurem
|-activate|p2a: Kyurem|move: Protect
|
|-weather|SunnyDay|[upkeep]
|-heal|p2a: Kyurem|39/100|[from] item: Leftovers
|upkeep
|turn|3
|j| TheGargFather
|j| LT111VZ Ox
|j| Micciu
|j| Tenebricite
|j| Separation
|j|@jetou
|
|t:|1722616765
|move|p1a: Ninetales|Encore|p2a: Kyurem
|-start|p2a: Kyurem|Encore
|move|p2a: Kyurem|Protect||[still]
|-fail|p2a: Kyurem
|
|-weather|SunnyDay|[upkeep]
|-heal|p2a: Kyurem|45/100|[from] item: Leftovers
|upkeep
|turn|4
|
|t:|1722616773
|switch|p1a: Walking Wake|Walking Wake|100/100
|-activate|p1a: Walking Wake|ability: Protosynthesis
|-start|p1a: Walking Wake|protosynthesisspe
|move|p2a: Kyurem|Protect||[still]
|-fail|p2a: Kyurem
|
|-weather|SunnyDay|[upkeep]
|-heal|p2a: Kyurem|51/100|[from] item: Leftovers
|upkeep
|turn|5
|
|t:|1722616783
|-end|p1a: Walking Wake|Protosynthesis|[silent]
|switch|p1a: Zamazenta|Zamazenta|100/100
|-ability|p1a: Zamazenta|Dauntless Shield|boost
|-boost|p1a: Zamazenta|def|1
|move|p2a: Kyurem|Protect||[still]
|-fail|p2a: Kyurem
|
|-weather|SunnyDay|[upkeep]
|-heal|p2a: Kyurem|57/100|[from] item: Leftovers
|-end|p2a: Kyurem|Encore
|upkeep
|turn|6
|
|t:|1722616787
|move|p2a: Kyurem|Protect|p2a: Kyurem
|-singleturn|p2a: Kyurem|Protect
|move|p1a: Zamazenta|Roar|p2a: Kyurem
|drag|p2a: Cinderace|Cinderace, M|100/100
|
|-weather|SunnyDay|[upkeep]
|upkeep
|turn|7
|l| Tenebricite
|
|t:|1722616791
|move|p1a: Zamazenta|Body Press|p2a: Cinderace
|-damage|p2a: Cinderace|20/100
|move|p2a: Cinderace|U-turn|p1a: Zamazenta
|-start|p2a: Cinderace|typechange|Bug|[from] ability: Libero
|-resisted|p1a: Zamazenta
|-damage|p1a: Zamazenta|92/100
|
|t:|1722616807
|switch|p2a: Zapdos|Zapdos|100/100|[from] U-turn
|
|-weather|SunnyDay|[upkeep]
|-heal|p1a: Zamazenta|98/100|[from] item: Leftovers
|upkeep
|turn|8
|j| mh=pb
|j| zawgshlawg
|j| baby gap jeans
|j| aketsuki
|j| chiang kai shrek
|j| almaluna
|j| WEDemon
|j| LeHellom James
|
|t:|1722616830
|move|p1a: Zamazenta|Body Press|p2a: Zapdos
|-resisted|p2a: Zapdos
|-damage|p2a: Zapdos|80/100
|move|p2a: Zapdos|Thunder Wave|p1a: Zamazenta
|-status|p1a: Zamazenta|par
|
|-weather|none
|-heal|p1a: Zamazenta|100/100 par|[from] item: Leftovers
|upkeep
|turn|9
|
|t:|1722616836
|switch|p1a: Sandy Shocks|Sandy Shocks|100/100
|move|p2a: Zapdos|Volt Switch|p1a: Sandy Shocks
|-immune|p1a: Sandy Shocks
|
|upkeep
|turn|10
|j| lt111vz suhayb
|j| radahndorus
|
|t:|1722616842
|move|p1a: Sandy Shocks|Earth Power|p2a: Zapdos
|-immune|p2a: Zapdos
|move|p2a: Zapdos|Roost|p2a: Zapdos
|-heal|p2a: Zapdos|100/100
|-singleturn|p2a: Zapdos|move: Roost
|
|upkeep
|turn|11
|
|t:|1722616844
|move|p1a: Sandy Shocks|Thunderbolt|p2a: Zapdos
|-damage|p2a: Zapdos|62/100
|move|p2a: Zapdos|Hurricane|p1a: Sandy Shocks
|-resisted|p1a: Sandy Shocks
|-damage|p1a: Sandy Shocks|71/100
|
|upkeep
|turn|12
|
|t:|1722616848
|-terastallize|p1a: Sandy Shocks|Ice
|move|p1a: Sandy Shocks|Tera Blast|p2a: Zapdos|[anim] Tera Blast Ice
|-supereffective|p2a: Zapdos
|-damage|p2a: Zapdos|0 fnt
|faint|p2a: Zapdos
|
|upkeep
|j| sv coinflipping
|j| xavgb
|j| hughduffy
|
|t:|1722616858
|switch|p2a: Cinderace|Cinderace, M|20/100
|turn|13
|
|t:|1722616866
|-end|p1a: Sandy Shocks|Protosynthesis|[silent]
|switch|p1a: Landorus|Landorus-Therian, M|100/100
|-ability|p1a: Landorus|Intimidate|boost
|-unboost|p2a: Cinderace|atk|1
|move|p2a: Cinderace|U-turn|p1a: Landorus
|-start|p2a: Cinderace|typechange|Bug|[from] ability: Libero
|-resisted|p1a: Landorus
|-damage|p1a: Landorus|89/100
|-damage|p2a: Cinderace|3/100|[from] item: Rocky Helmet|[of] p1a: Landorus
|j| t_yeu
|
|t:|1722616869
|switch|p2a: Kyurem|Kyurem|57/100|[from] U-turn
|-ability|p2a: Kyurem|Pressure
|
|-heal|p2a: Kyurem|63/100|[from] item: Leftovers
|upkeep
|turn|14
|
|t:|1722616873
|switch|p1a: Ninetales|Ninetales, F|100/100
|-weather|SunnyDay|[from] ability: Drought|[of] p1a: Ninetales
|move|p2a: Kyurem|Freeze-Dry|p1a: Ninetales
|-resisted|p1a: Ninetales
|-damage|p1a: Ninetales|81/100
|
|-weather|SunnyDay|[upkeep]
|-heal|p2a: Kyurem|69/100|[from] item: Leftovers
|upkeep
|turn|15
|j| LT111VZ Mav
|n| dragonitefan58|svcoinflipping
|
|t:|1722616881
|switch|p1a: Walking Wake|Walking Wake|100/100
|-activate|p1a: Walking Wake|ability: Protosynthesis
|-start|p1a: Walking Wake|protosynthesisspe
|move|p2a: Kyurem|Earth Power|p1a: Walking Wake
|-damage|p1a: Walking Wake|59/100
|
|-weather|SunnyDay|[upkeep]
|-heal|p2a: Kyurem|75/100|[from] item: Leftovers
|upkeep
|turn|16
|
|t:|1722616892
|move|p1a: Walking Wake|Hydro Steam|p2a: Kyurem
|-resisted|p2a: Kyurem
|-damage|p2a: Kyurem|41/100
|move|p2a: Kyurem|Earth Power|p1a: Walking Wake
|-damage|p1a: Walking Wake|21/100
|
|-weather|SunnyDay|[upkeep]
|-heal|p2a: Kyurem|47/100|[from] item: Leftovers
|upkeep
|turn|17
|l| WEDemon
|
|t:|1722616899
|move|p2a: Kyurem|Protect|p2a: Kyurem
|-singleturn|p2a: Kyurem|Protect
|move|p1a: Walking Wake|Hydro Steam|p2a: Kyurem
|-activate|p2a: Kyurem|move: Protect
|
|-weather|SunnyDay|[upkeep]
|-heal|p2a: Kyurem|53/100|[from] item: Leftovers
|upkeep
|turn|18
|l| lt111vz suhayb
|j| qd king
|inactive|LT111VZ bewashed has 120 seconds left.
|l| Sujay532
|j| 12ad12
|j| LT111VZ JD
|
|t:|1722616950
|move|p1a: Walking Wake|Dragon Pulse|p2a: Kyurem
|-supereffective|p2a: Kyurem
|-damage|p2a: Kyurem|0 fnt
|faint|p2a: Kyurem
|
|-weather|SunnyDay|[upkeep]
|upkeep
|
|t:|1722616958
|switch|p2a: Roaring Moon|Roaring Moon|100/100
|-activate|p2a: Roaring Moon|ability: Protosynthesis
|-start|p2a: Roaring Moon|protosynthesisatk
|turn|19
|j| Jenthai
|
|t:|1722616965
|-end|p1a: Walking Wake|Protosynthesis|[silent]
|switch|p1a: Landorus|Landorus-Therian, M|89/100
|-ability|p1a: Landorus|Intimidate|boost
|-unboost|p2a: Roaring Moon|atk|1
|move|p2a: Roaring Moon|U-turn|p1a: Landorus
|-resisted|p1a: Landorus
|-damage|p1a: Landorus|79/100
|-damage|p2a: Roaring Moon|84/100|[from] item: Rocky Helmet|[of] p1a: Landorus
|
|t:|1722616969
|-end|p2a: Roaring Moon|Protosynthesis|[silent]
|switch|p2a: Great Tusk|Great Tusk|100/100|[from] U-turn
|-activate|p2a: Great Tusk|ability: Protosynthesis
|-start|p2a: Great Tusk|protosynthesisatk
|
|-weather|SunnyDay|[upkeep]
|upkeep
|turn|20
|j| sample nerd
|
|t:|1722616977
|move|p1a: Landorus|U-turn|p2a: Great Tusk
|-resisted|p2a: Great Tusk
|-damage|p2a: Great Tusk|93/100
|
|t:|1722616979
|switch|p1a: Sandy Shocks|Sandy Shocks, tera:Ice|71/100|[from] U-turn
|-activate|p1a: Sandy Shocks|ability: Protosynthesis
|-start|p1a: Sandy Shocks|protosynthesisspe
|move|p2a: Great Tusk|Rapid Spin|p1a: Sandy Shocks
|-damage|p1a: Sandy Shocks|48/100
|-boost|p2a: Great Tusk|spe|1
|
|-weather|SunnyDay|[upkeep]
|upkeep
|turn|21
|j|+AM
|inactive|LT111VZ bewashed has 120 seconds left.
|
|t:|1722617002
|-terastallize|p2a: Great Tusk|Fighting
|move|p1a: Sandy Shocks|Tera Blast|p2a: Great Tusk|[anim] Tera Blast Ice
|-damage|p2a: Great Tusk|34/100
|move|p2a: Great Tusk|Close Combat|p1a: Sandy Shocks
|-supereffective|p1a: Sandy Shocks
|-damage|p1a: Sandy Shocks|0 fnt
|-unboost|p2a: Great Tusk|def|1
|-unboost|p2a: Great Tusk|spd|1
|faint|p1a: Sandy Shocks
|-end|p1a: Sandy Shocks|Protosynthesis|[silent]
|
|-weather|none
|-end|p2a: Great Tusk|Protosynthesis
|upkeep
|
|t:|1722617007
|switch|p1a: Landorus|Landorus-Therian, M|79/100
|-ability|p1a: Landorus|Intimidate|boost
|-unboost|p2a: Great Tusk|atk|1
|turn|22
|
|t:|1722617014
|switch|p1a: Gouging Fire|Gouging Fire|100/100
|move|p2a: Great Tusk|Ice Spinner|p1a: Gouging Fire
|-damage|p1a: Gouging Fire|85/100
|
|upkeep
|turn|23
|j| iron violent
|
|t:|1722617023
|move|p2a: Great Tusk|Headlong Rush|p1a: Gouging Fire
|-supereffective|p1a: Gouging Fire
|-damage|p1a: Gouging Fire|19/100
|-unboost|p2a: Great Tusk|def|1
|-unboost|p2a: Great Tusk|spd|1
|move|p1a: Gouging Fire|Heat Crash|p2a: Great Tusk
|-damage|p2a: Great Tusk|0 fnt
|faint|p2a: Great Tusk
|-end|p2a: Great Tusk|Protosynthesis|[silent]
|
|upkeep
|c| iron violent|dang LT111VZ bewashed its not looking too hot for you
|j| KantoRemakeWhen
|
|t:|1722617042
|switch|p2a: Cinderace|Cinderace, M|3/100
|turn|24
|l| hughduffy
|j| hughduffy
|inactive|LT111VZ bewashed has 120 seconds left.
|
|t:|1722617062
|move|p2a: Cinderace|Gunk Shot|p1a: Gouging Fire
|-start|p2a: Cinderace|typechange|Poison|[from] ability: Libero
|-damage|p1a: Gouging Fire|0 fnt
|faint|p1a: Gouging Fire
|-end|p1a: Gouging Fire|Protosynthesis|[silent]
|
|upkeep
|l| xuanse123
|
|t:|1722617070
|switch|p1a: Landorus|Landorus-Therian, M|79/100
|-ability|p1a: Landorus|Intimidate|boost
|-unboost|p2a: Cinderace|atk|1
|turn|25
|c| iron violent|ninetales looks so cool
|j| LEAN DREAMS
|c|☆LT111VZ bewashed|It’s alright we gushi
|j| LT111VZ spooky
|inactive|LT111VZ bewashed has 120 seconds left.
|j| lt111vzpo
|
|t:|1722617090
|move|p2a: Cinderace|Pyro Ball|p1a: Landorus
|-damage|p1a: Landorus|55/100
|-status|p1a: Landorus|brn
|move|p1a: Landorus|U-turn|p2a: Cinderace
|-resisted|p2a: Cinderace
|-damage|p2a: Cinderace|0 fnt
|faint|p2a: Cinderace
|j| DangerToZU
|
|t:|1722617097
|switch|p1a: Ninetales|Ninetales, F|81/100|[from] U-turn
|-weather|SunnyDay|[from] ability: Drought|[of] p1a: Ninetales
|
|-weather|SunnyDay|[upkeep]
|upkeep
|j| LT111VZ Light
|
|t:|1722617105
|switch|p2a: Roaring Moon|Roaring Moon|84/100
|-activate|p2a: Roaring Moon|ability: Protosynthesis
|-start|p2a: Roaring Moon|protosynthesisatk
|turn|26
|j| OLT GHOSTER
|j| TalonSFlames
|inactive|LT111VZ bewashed has 120 seconds left.
|l| TalonSFlames
|j| LT111VZ Lacks
|
|t:|1722617148
|switch|p1a: Landorus|Landorus-Therian, M|55/100 brn
|-ability|p1a: Landorus|Intimidate|boost
|-unboost|p2a: Roaring Moon|atk|1
|move|p2a: Roaring Moon|Knock Off|p1a: Landorus
|-damage|p1a: Landorus|8/100 brn
|-damage|p2a: Roaring Moon|67/100|[from] item: Rocky Helmet|[of] p1a: Landorus
|-enditem|p1a: Landorus|Rocky Helmet|[from] move: Knock Off|[of] p2a: Roaring Moon
|
|-weather|SunnyDay|[upkeep]
|-damage|p1a: Landorus|2/100 brn|[from] brn
|upkeep
|turn|27
|j| waywirewaifu
|j| TalonSFlames
|c| OLT GHOSTER|hey bewashed
|
|t:|1722617157
|move|p2a: Roaring Moon|Knock Off|p1a: Landorus
|-damage|p1a: Landorus|0 fnt
|faint|p1a: Landorus
|
|-weather|SunnyDay|[upkeep]
|upkeep
|
|t:|1722617161
|switch|p1a: Ninetales|Ninetales, F|81/100
|turn|28
|
|t:|1722617176
|-end|p2a: Roaring Moon|Protosynthesis|[silent]
|switch|p2a: Tinkaton|Tinkaton, F|100/100
|-ability|p2a: Tinkaton|Mold Breaker
|move|p1a: Ninetales|Healing Wish|p1a: Ninetales
|faint|p1a: Ninetales
|
|-weather|SunnyDay|[upkeep]
|upkeep
|
|t:|1722617180
|switch|p1a: Zamazenta|Zamazenta|100/100 par
|-heal|p1a: Zamazenta|100/100|[from] move: Healing Wish
|turn|29
|
|t:|1722617193
|move|p1a: Zamazenta|Iron Defense|p1a: Zamazenta
|-boost|p1a: Zamazenta|def|2
|move|p2a: Tinkaton|Encore|p1a: Zamazenta
|-start|p1a: Zamazenta|Encore
|
|-weather|SunnyDay|[upkeep]
|upkeep
|turn|30
|
|t:|1722617196
|switch|p2a: Roaring Moon|Roaring Moon|67/100
|-activate|p2a: Roaring Moon|ability: Protosynthesis
|-start|p2a: Roaring Moon|protosynthesisatk
|move|p1a: Zamazenta|Iron Defense|p1a: Zamazenta
|-boost|p1a: Zamazenta|def|2
|
|-weather|SunnyDay|[upkeep]
|upkeep
|turn|31
|l| xavgb
|
|t:|1722617198
|move|p2a: Roaring Moon|Outrage|p1a: Zamazenta
|-damage|p1a: Zamazenta|77/100
|move|p1a: Zamazenta|Iron Defense|p1a: Zamazenta
|-boost|p1a: Zamazenta|def|2
|
|-weather|SunnyDay|[upkeep]
|-heal|p1a: Zamazenta|83/100|[from] item: Leftovers
|upkeep
|turn|32
|
|t:|1722617201
|move|p2a: Roaring Moon|Outrage|p1a: Zamazenta|[from]lockedmove
|-damage|p1a: Zamazenta|65/100
|-start|p2a: Roaring Moon|confusion|[fatigue]
|move|p1a: Zamazenta|Iron Defense|p1a: Zamazenta
|-boost|p1a: Zamazenta|def|0
|
|-weather|none
|-end|p2a: Roaring Moon|Protosynthesis
|-heal|p1a: Zamazenta|71/100|[from] item: Leftovers
|-end|p1a: Zamazenta|Encore
|upkeep
|turn|33
|
|t:|1722617204
|-activate|p2a: Roaring Moon|confusion
|-damage|p2a: Roaring Moon|50/100|[from] confusion
|move|p1a: Zamazenta|Body Press|p2a: Roaring Moon
|-supereffective|p2a: Roaring Moon
|-damage|p2a: Roaring Moon|0 fnt
|faint|p2a: Roaring Moon
|-end|p2a: Roaring Moon|Protosynthesis|[silent]
|
|-heal|p1a: Zamazenta|77/100|[from] item: Leftovers
|upkeep
|j| GoodGameMetagame
|c|☆LT111VZ bewashed|Gg
|c|☆GALAXY☆00000|gg
|c| iron violent|zamazenta is broken
|l| 12ad12
|c|☆GALAXY☆00000|ikr
|
|t:|1722617222
|switch|p2a: Tinkaton|Tinkaton, F|100/100
|-ability|p2a: Tinkaton|Mold Breaker
|turn|34
|c| lt111vz smoolic|zamazenta mu is bad
|-message|LT111VZ bewashed forfeited.
|
|win|GALAXY☆00000
|raw|GALAXY☆00000's rating: 2128 → <strong>2148</strong><br />(+20 for winning)
|raw|LT111VZ bewashed's rating: 2129 → <strong>2109</strong><br />(-20 for losing)
|l|☆LT111VZ bewashed
|player|p2|
"""

# --- Run the Parser ---
parsed_states, final_state = parse_showdown_replay(replay_log, replay_id="example_battle_001")

# --- Flatten the States ---
flattened_data = [flatten_state(s) for s in parsed_states]

# --- Create DataFrame ---
df = pd.DataFrame(flattened_data)

# --- Display Example Output ---
pd.set_option('display.max_columns', None) # Show all columns
# pd.set_option('display.max_rows', 100) # Show more rows if needed

print(f"Total states captured: {len(df)}")
if not df.empty:
    print("\nFirst few rows of the DataFrame:")
    print(df.head())

    print("\nExample row (Turn 21, P1 Sandy Shocks about to move):")
    # Find the state: Turn 21, P1 moving, action was 'move:Tera Blast'
    example_row = df[(df['turn_number'] == 21) & (df['player_to_move'] == 'p1') & (df['action_taken'] == 'move:Tera Blast')]
    if not example_row.empty:
        print(example_row.iloc[0]) # Print the first match Series
    else:
        print("Could not find the specific example row T21 P1 'move:Tera Blast'.")

    print("\nLast few rows of the DataFrame:")
    print(df.tail())
else:
    print("No states were successfully parsed.")

# You can now save this DataFrame:
# df.to_csv('parsed_replay.csv', index=False)
# df.to_parquet('parsed_replay.parquet', index=False) # Recommended for large datasets