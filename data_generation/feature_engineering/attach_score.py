import os
import pandas as pd
import numpy as np
import itertools
print(os.getcwd())
import sys
sys.path.append(os.getcwd())

from feature_engineering.utils import most_common_element, notNone

WEAPON_TYPES = ['AR', 'DMR', 'LMG', 'SMG', 'SR']

THROWABLE_WEAPONS = [
    'Item_Weapon_SmokeBomb_C', 
    'Item_Weapon_FlashBang_C', 
    'Item_Weapon_Grenade_C', 
    'Item_Weapon_Molotov_C', 
    'Item_Weapon_BluezoneGrenade_C', 
    'Item_Weapon_Pan_C', 
    'Item_Weapon_PanzerFaust100M_C']


CAREPACKAGE_WEAPONS = [
    "Item_Weapon_P90_C",
    "Item_Weapon_Groza_C",
    "Item_Weapon_FamasG2_C",  # 대소문자 수정: FAMASG2 → FamasG2
    "Item_Weapon_MG3_C",
]

AR_WEAPONS = [
    "Item_Weapon_Mk12_C",
    "Item_Weapon_AUG_C",
    "Item_Weapon_BerylM762_C",
    "Item_Weapon_HK416_C",
    "Item_Weapon_ACE32_C",
    "Item_Weapon_SCAR-L_C",
    "Item_Weapon_AK47_C",
    "Item_Weapon_M16A4_C",
    "Item_Weapon_Mk47Mutant_C",
    "Item_Weapon_Groza_C",      # CAREPACKAGE AR
    "Item_Weapon_FamasG2_C",    # CAREPACKAGE AR
]
DMR_WEAPONS = [
    "Item_Weapon_Mini14_C",
    "Item_Weapon_SKS_C",
    "Item_Weapon_VSS_C",
    "Item_Weapon_Mk14_C",
    "Item_Weapon_FNFal_C",
    "Item_Weapon_Dragunov_C",
]
SR_WEAPONS = [
    "Item_Weapon_M24_C",
    "Item_Weapon_Kar98k_C",
    "Item_Weapon_Mosin_C",
    "Item_Weapon_AWM_C",
]
SMG_WEAPONS = [
    "Item_Weapon_Vector_C",
    "Item_Weapon_MP5K_C",
    "Item_Weapon_UMP_C",
    "Item_Weapon_UZI_C",
    "Item_Weapon_vz61Skorpion_C",
    "Item_Weapon_BizonPP19_C",
    "Item_Weapon_P90_C",        # CAREPACKAGE SMG
]
LMG_WEAPONS = [
    "Item_Weapon_M249_C",      # LMG
    "Item_Weapon_DP28_C",      # LMG
    "Item_Weapon_MG3_C",       # CAREPACKAGE LMG
]
SHOTGUN_WEAPONS = [
     "Item_Weapon_DP12_C",      # Shotgun DBS
     "Item_Weapon_Saiga12_C",   # Shotgun
     "Item_Weapon_Berreta686_C",# Shotgun
     "Item_Weapon_Winchester_C",# Shotgun S1897
]

EXCEPT_MAINS = ['Item_Weapon_FlareGun_C', 'Item_Weapon_PanzerFaust100M_C', 'Item_Weapon_Mortar_C', 'Item_Weapon_TacPack_C', 'Item_Weapon_IntegratedRepair_C', 'Item_Weapon_TraumaBag_C']
ATTACH_TYPES = ["Muzzle", "Lower", "Upper", "Magazine", "Stock"]


AR_ATTACH_SCORES = dict()
AR_ATTACH_SCORES['Muzzle'] = {'BASE':6}
AR_ATTACH_SCORES['Magazine'] = {'BASE':4}
AR_ATTACH_SCORES['Lower'] = {'BASE':3}
AR_ATTACH_SCORES['Stock'] = {'BASE':3}
AR_ATTACH_SCORES['Upper'] = {'BASE':2}

DMR_ATTACH_SCORES = dict()
DMR_ATTACH_SCORES['Muzzle'] = {'BASE': 5}
DMR_ATTACH_SCORES['Magazine'] = {'BASE':4}
DMR_ATTACH_SCORES['Lower'] = {'BASE':4}
DMR_ATTACH_SCORES['Stock'] = {'BASE':4}
DMR_ATTACH_SCORES['Upper'] = {'BASE':7}

SR_ATTACH_SCORES = dict()
SR_ATTACH_SCORES['Muzzle'] = {'BASE':1,
                              'Item_Attach_Weapon_Muzzle_Suppressor_SniperRifle_C':6,
                              'Item_Attach_Weapon_Muzzle_Compensator_SniperRifle_C':0,
                              'Item_Attach_Weapon_Muzzle_FlashHider_SniperRifle_C':2}
SR_ATTACH_SCORES['Magazine'] = {'BASE':2}
SR_ATTACH_SCORES['Stock'] = {'BASE':3}
SR_ATTACH_SCORES['Upper'] = {'BASE':8}



def find_main_weapon(row, item_list, target_item, start_t, end_t):
    phase1_pickup = [(pt, pi) for pt, pi in zip(row.pickup_time, row.pickup_item) if (start_t <= pt) & (pt < end_t)]
    phase1_good_items = [(pt, pi) for pt, pi in phase1_pickup if pi in item_list]
    if len(phase1_good_items) == 0:
        return None
    
    last_pt, last_pi = phase1_good_items[-1]
    if last_pi in target_item:
        return last_pi
    else:
        return None
    

def deprecated_calculate_attach_score(row, attach_score_def):
    score_dict = dict()
    
    for pi in row.pickup_item:
        # print(pi)
        if len(pi.split('_')) < 4:
            continue
        part_type = pi.split('_')[3]
        
        if part_type in attach_score_def.keys():
            if pi in attach_score_def[part_type].keys():
                score_dict[part_type] = attach_score_def[part_type][pi]
                
    return score_dict

def calculate_attach_score(row, attach_score_def, end_time, start_time=1):
    if notNone(row.attach_time):
        pass
    else:
        return None
    
    if row.main_weapon in AR_WEAPONS:
        base_score_def = AR_ATTACH_SCORES
    elif row.main_weapon in DMR_WEAPONS:
        base_score_def = DMR_ATTACH_SCORES
    elif row.main_weapon in SR_WEAPONS:
        base_score_def = SR_ATTACH_SCORES
    else:
        raise Exception(f'{row.main_weapon} Not Supported')
    
    score_dict = dict()
    if row.main_weapon not in attach_score_def.keys():
        return None
    
    weapon_attach_score_def = attach_score_def[row.main_weapon]
    
    for at, ap, ac, ac_cate in zip(row.attach_time, row.attach_parent, row.attach_child, row.attach_child_cate):
        if (ap == row.main_weapon) & (at < end_time) & (ac_cate != 'SideRail') & (ac_cate in weapon_attach_score_def.keys()):
            attach_def = weapon_attach_score_def[ac_cate]
            if ac in attach_def.keys():
                score_dict[ac_cate] = attach_def[ac] * (base_score_def[ac_cate][ac] if ac in base_score_def[ac_cate].keys() else base_score_def[ac_cate]['BASE']) 
            else:
                score_dict[ac_cate] = 0
            
    return score_dict


def find_preferred_weapons_in_matches(df, threshold=0.95):
    last_weapons = df.apply(lambda row: [pi for pt, pi, pc in zip(row.pickup_time, row.pickup_item, row.pickup_cate) 
                                            if (pc == 'Main') & (pt > 900) & (pi not in EXCEPT_MAINS)][-2:] 
                                            if notNone(row.pickup_time) else [], axis=1).tolist()

    picked_weapons = list(itertools.chain(*last_weapons))
    n_picked_weapons = len(picked_weapons)
    frequently_used_weapons = pd.Series(picked_weapons).value_counts() / n_picked_weapons
    # print(frequently_used_weapons)
    
    cumulative_sum = 0
    selected_weapons = []

    for weapon, usage in frequently_used_weapons.items():
        cumulative_sum += usage
        selected_weapons.append(weapon)
        if cumulative_sum >= threshold:
            break
        
    return selected_weapons


def calculate_weapon_score_per_attach(df, item_name='Item_Weapon_Dragunov_C'):
    df['death_time'] = df['death_time'].fillna(2000)
    weapon_attach = df.apply(lambda row: [ac for at, ap, ac in zip(row.attach_time, row.attach_parent, row.attach_child)
                                        if ap == item_name]
                                        if notNone(row.attach_time) else [], axis=1).tolist()
    
    weapon_attach = list(itertools.chain(*weapon_attach))
    weapon_attach_df = pd.DataFrame(weapon_attach, columns=['attach_name'])
    weapon_attach_df['attach_cate'] = weapon_attach_df.attach_name.apply(lambda x: x.split('_')[3] if x and len(x.split('_')) > 3 else None)
    possible_attach_types = weapon_attach_df.attach_cate.value_counts().index.tolist()
    
    all_weapon_attach_score = df.apply(lambda row: [(at, ac, ac_cate) for at, ap, ac, ac_cate in zip(row.attach_time, row.attach_parent, row.attach_child, row.attach_child_cate)
                                                if (ap == item_name)] 
                                                if notNone(row.attach_time) else [], axis=1)
    

    attach_score = dict()
    for attach_type in possible_attach_types:
        if attach_type == 'SideRail':
            continue

        weapon_attach_score = all_weapon_attach_score.apply(lambda x: [(at, ac) for at, ac, ac_cate in x if ac_cate == attach_type])
        weapon_attach_score = weapon_attach_score.apply(lambda x: ([ac for at, ac in x if at > 900]) 
                                                        if len(x) > 1 else [])
        
        weapon_attach_score = weapon_attach_score.tolist()
        weapon_attach_score_df = itertools.chain(*weapon_attach_score)
        weapon_attach_score_df = pd.DataFrame(weapon_attach_score_df, columns=['attach_name'])
        weapon_attach_score_df['attach_cate'] = weapon_attach_score_df.attach_name.apply(lambda x: x.split('_')[3] if x and len(x.split('_')) > 3 else None)
        
        attach_logs = weapon_attach_score_df[weapon_attach_score_df.attach_cate == attach_type]
        n_attach_logs = len(attach_logs)
        # print(f'{attach_type}, {n_attach_logs}')
        if n_attach_logs < 10:
            if item_name in CAREPACKAGE_WEAPONS:
                attach_score[attach_type] = 1.0
            continue

        n_types_attach = 4

        attach_counts = attach_logs.attach_name.value_counts()[:n_types_attach]
        attach_counts = attach_counts/attach_counts.sum()
        attach_score[attach_type] = attach_counts.round(2).to_dict()
        
    return attach_score