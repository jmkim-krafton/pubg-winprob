PHASES_START = [0, 90, 600, 810, 990, 1170, 1350, 1530, 1680, 1800, 1970]
PHASES_MOVE = [0, 330, 690, 870, 1050, 1230, 1410, 1590, 1740, 1810, 2000]
PHASES_END = [90, 600, 810, 990, 1170, 1350, 1530, 1680, 1800, 1970, 2000]
PHASES_TIME = [0, 330, 690, 870, 1050, 1230, 1410, 1590, 1740, 1810, 2000]

FEATURES0 = dict()
FEATURES0['phase1_features'] = ['phase1_n_pickup', 'phase1_n_door', 'phase1_n_drop', 'phase1_AR_attach_score', 'phase1_SR_attach_score', 'phase1_DMR_attach_score']
FEATURES0['match_features'] = ['death_time', 'total_n_attack']
FEATURES0['category_features'] = ['phase1_pickup_city','phase1_fight4stat']  
FEATURES0['state_features'] = ['phase1_pickup_city', 'phase1_incircle', 'phase1_fight4stat']

FEATURES = [FEATURES0]

FEATURES1_PER_PHASE = [
    'n_pickup',
    'n_door',
    'n_drop',
    'n_attack',
    'n_teammates',
    'n_enemies',
    'd_incircle',
    # 'rooting_city',
    'score_AR_attach',
    'score_DMR_attach',
    'score_SR_attach',
    'time_Npickup',
    'dist_Npickup',
    'dist_Npickup_vehicle',
    'n_AR_attack',
    'n_SR_attack',
    'n_DMR_attack',                                     
    'n_SMG_attack',
    'n_SHOTGUN_attack',
    'n_dmg_by_AR',
    'mean_dist_by_AR',
    'sum_dmg_by_AR',
    'n_dmg_by_SR',
    'mean_dist_by_SR',
    'sum_dmg_by_SR',
    'n_dmg_by_DMR',
    'mean_dist_by_DMR',
    'sum_dmg_by_DMR',
    'n_dmg_by_SMG',
    'mean_dist_by_SMG',
    'sum_dmg_by_SMG',
    'n_dmg_by_SHOTGUN',
    'mean_dist_by_SHOTGUN',
    'sum_dmg_by_SHOTGUN',
    'n_takedmg_by_AR',
    'mean_takedmg_dist_by_AR',
    'sum_takedmg_by_AR',
    'n_takedmg_by_SR',
    'mean_takedmg_dist_by_SR',
    'sum_takedmg_by_SR',
    'n_takedmg_by_DMR',
    'mean_takedmg_dist_by_DMR',
    'sum_takedmg_by_DMR',
    'n_takedmg_by_SMG',
    'mean_takedmg_dist_by_SMG',
    'sum_takedmg_by_SMG',
    'n_takedmg_by_SHOTGUN',
    'mean_takedmg_dist_by_SHOTGUN',
    'sum_takedmg_by_SHOTGUN',
    'n_kill',
    'n_finish',
    'n_groggy',
    'n_squad_last_member_kill',
    'n_camping',
    'n_noaction_camping',
    'n_attack_camping',
    'n_pickup_camping',
    'n_groggy_camping'
]


SEASONS = {
    'season15': ('2021-11-30', '2022-02-16'),
    'season16': ('2022-02-16', '2022-04-13'),
    'season17': ('2022-04-13', '2022-06-08'),
    'season18': ('2022-06-08', '2022-08-09'),
    'season19': ('2022-08-09', '2022-10-12'),
    'season20': ('2022-10-12', '2022-12-06'),
    'season21': ('2022-12-06', '2023-02-15'),
    'season22': ('2023-02-15', '2023-04-12'),
    'season23': ('2023-04-12', '2023-06-14'),
    'season24': ('2023-06-14', '2023-08-09'),
    'season25': ('2023-08-09', '2023-10-11'),
    'season26': ('2023-10-11', '2023-12-06'),
    'season27': ('2023-12-06', '2024-02-07'),
    'season28': ('2024-02-07', '2024-04-09'),
    'season29': ('2024-04-09', '2024-06-12'),
    'season30': ('2024-06-12', '2024-08-07'),
    'season31': ('2024-08-07', '2024-10-10'),
    'season32': ('2024-10-10', '2024-12-04'),
    'season33': ('2024-12-04', '2025-02-12'),
    'season34': ('2024-02-12', '2025-04-09'),
}

# MAP_NAME_TO_LEGACY = {
#     "miramar": "Desert_Main",      #Mirama
#     "taego": "Tiger_Main",       #Taego
#     "rondo": "Neon_Main",        #Rondo
#     "erangel": "Baltic_Main",      #Erangel
#     "deston": "Kiki_Main",        #Deston
#     "vikendi": "DihorOtok_Main",   #Vikendi
# }

MAP_NAME_TO_LEGACY = {
"Desert_Main": "miramar", #Mirama
"Tiger_Main": "taego",  #Taego
"Neon_Main": "rondo",   #Rondo
"Baltic_Main": "erangel", #Erangel
"Kiki_Main": "deston",   #Deston
"DihorOtok_Main":"vikendi",   #Vikendi
}

MAP_NAME_TO_LEGACY_OFFICIAL = {
    "sanhok": "Savage_Main",    #Sanhok
    "karakin": "Summerland_Main",   #Karakin
    "miramar": "Desert_Main",  #Mirama
    "taego_official": "Tiger_Main",       #Taego
    "rondo": "Neon_Main",        #Rondo
    "erangel": "Baltic_Main",      #Erangel
    "deston": "Kiki_Main",        #Deston
    "vikendi": "DihorOtok_Main",   #Vikendi
    "paramo": "Chimera_Main"    #Paramo
}

# 🎯 Item Mapping for new features
HELMET_ITEMS = {
    'Item_Head_E_01_Lv1_C': 1,
    'Item_Head_E_02_Lv1_C': 1,
    'Item_Head_F_01_Lv2_C': 2,
    'Item_Head_F_02_Lv2_C': 2,
    'Item_Head_G_01_Lv3_C': 3,
    'Item_Head_G_02_Lv3_C': 3,
}

VEST_ITEMS = {
    'Item_Armor_E_01_Lv1_C': 1,
    'Item_Armor_D_01_Lv2_C': 2,
    'Item_Armor_C_01_Lv3_C': 3,
}

BACKPACK_ITEMS = {
    'Item_Back_E_01_Lv1_C': 1,
    'Item_Back_E_02_Lv1_C': 1,
    'Item_Back_F_01_Lv2_C': 2,
    'Item_Back_F_02_Lv2_C': 2,
    'Item_Back_C_01_Lv3_C': 3,
    'Item_Back_C_02_Lv3_C': 3,
}

THROWABLE_ITEMS = {
    'Item_Weapon_Grenade_C': 'grenade',
    'Item_Weapon_SmokeBomb_C': 'smoke',
    'Item_Weapon_FlashBang_C': 'flashbang',
    'Item_Weapon_Molotov_C': 'molotov',
    'Item_Weapon_StunGun_C': 'stun',
    'Item_Weapon_BluezoneGrenade_C': 'bluezone_grenade',
}

AIDS_ITEMS = {
    'Item_Heal_Bandage_C': 'bandage',
    'Item_Heal_FirstAid_C': 'firstaid',
    'Item_Heal_MedKit_C': 'medkit',
    'Item_Boost_PainKiller_C': 'painkiller',
    'Item_Boost_EnergyDrink_C': 'energydrink',
    'Item_Boost_AdrenalineSyringe_C': 'adrenaline',
}