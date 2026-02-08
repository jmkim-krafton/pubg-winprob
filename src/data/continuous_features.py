"""
Continuous feature definitions for PGC dataset.
"""

CONTINUOUS_FEATURES = [
    # Action Cumulative Features (28)
    'n_pickup',
    'n_drop',
    'n_kill',
    'n_finish',
    'n_groggy',
    'n_dmg',
    'n_dmg_by_AR',
    'n_dmg_by_SR',
    'n_dmg_by_DMR',
    'n_dmg_by_SMG',
    'n_dmg_by_SHOTGUN',
    'sum_dmg',
    'sum_dmg_by_AR',
    'sum_dmg_by_SR',
    'sum_dmg_by_DMR',
    'sum_dmg_by_SMG',
    'sum_dmg_by_SHOTGUN',
    'n_take_dmg',
    'sum_take_dmg',
    'n_heal',
    'sum_heal_amount',
    'n_heal_bandage',
    'n_heal_firstaid',
    'n_heal_medkit',
    'n_heal_painkiller',
    'n_heal_energydrink',
    'n_heal_adrenaline',
    'n_vehicle_ride',

    # Snapshot Features - Location (4)
    'n_teammates',
    'n_enemies',
    'dist_from_bluezone',
    'dist_from_whitezone',
    'dist_from_bluezone_v2',
    'dist_from_whitezone_v2',

    # Snapshot Features - Vehicle (1)
    'is_vehicle_count',

    # Snapshot Features - Armor/Backpack (3)
    'item_helmet_level',
    'item_vest_level',
    'item_backpack_level',

    # Snapshot Features - Throwables (6)
    'item_throwables_count',
    'item_grenade_count',
    'item_smoke_count',
    'item_flashbang_count',
    'item_molotov_count',
    'item_bluezone_grenade_count',

    # Snapshot Features - Aids (7)
    'item_aids_count',
    'item_bandage_count',
    'item_firstaid_count',
    'item_medkit_count',
    'item_painkiller_count',
    'item_energydrink_count',
    'item_adrenaline_count',

    # Snapshot Features - Team Status (4)
    'squad_total_health',
    'squad_alive_count',
    'squad_non_groggy_count',
    'squad_out_bluezone_count',
]


