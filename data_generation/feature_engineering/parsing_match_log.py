import os
import time
import pandas as pd
import numpy as np
import itertools
import concurrent.futures

print(os.getcwd())
import sys
sys.path.append(os.getcwd())

from feature_engineering.utils import extract_MatchId_from, load_match_df_from_ded_jsonl, load_match_df_from_s3, dict2cols, notNone, calculate_distance, most_common_element, top_k_frequent_items
from feature_engineering.match_table_info import target_table_and_cols

decorator_output = False
ACCOUNT_ID_LENGTH = 40

def time_checker(func):
    def wrapper(*args, **kwargs):
        start_time = time.time() 
        result = func(*args, **kwargs)  
        end_time = time.time()
        if decorator_output:
            print(f"{func.__name__}: {end_time - start_time:.4f} seconds")
        return result
    return wrapper


class MatchLogParser:
    def __init__(self, match_log_json_path, source='openapi'):
        if source == 'openapi':
            match_df, match_id, N = extract_MatchId_from(match_log_json_path, return_df=True)
        else:
            use_cols = list(set(list(itertools.chain(*[v for k, v in target_table_and_cols.items()]))))
            match_df, match_id, N = load_match_df_from_s3(match_log_json_path, target_table_and_cols.keys(), use_cols, return_df=True)
        
        self.match_id = match_id
        self.match_df = match_df
        self.match_df.columns = self.match_df.columns.str.lower()
        self.match_df._t = self.match_df._t.str.lower()
        self.match_df = self.normalize_ingame_time(self.match_df)
        self.zone_info = self.parse_zone_info(self.match_df)  # 🎯 SafeZone(블루존), PoisonGasWarning(화이트존) 정보 한 번만 파싱
        self.user_match_str_len = len(self.match_id) + 1 + ACCOUNT_ID_LENGTH # For filtering wierd account_id (1 is for '@')
        self.character_order = self._parse_character_order_from_match_start()  # 🎯 LogMatchStart 기준 character 순서
    
    def _parse_character_order_from_match_start(self):
        """
        LogMatchStart의 characters 배열 순서를 파싱하여 팀별 character 순서 저장
        Returns: {teamId: [accountId1, accountId2, ...], ...}
        """
        character_order = {}
        logmatchstart = self.match_df[self.match_df['_t'] == 'logmatchstart']
        
        if logmatchstart.empty:
            return character_order
        
        # characters 컬럼 접근 (안전하게 처리)
        first_row = logmatchstart.iloc[0]
        if 'characters' not in first_row.index:
            return character_order
        
        characters = first_row['characters']
        if characters is None or (isinstance(characters, list) and len(characters) == 0):
            return character_order
        
        # characters 배열을 순서대로 순회하면서 팀별로 저장
        for char_info in characters:
            if isinstance(char_info, dict) and 'character' in char_info:
                char = char_info['character']
                team_id = char.get('teamId')
                account_id = char.get('accountId')
                
                if team_id is not None and account_id:
                    if team_id not in character_order:
                        character_order[team_id] = []
                    character_order[team_id].append(account_id)
        
        return character_order

    def normalize_ingame_time(self, match_df) -> pd.DataFrame:
        if ('event_at' in match_df.columns) & (match_df['_d'].apply(len).iloc[0] < 12):
            match_df['_d'] = match_df['event_at']

        match_df['_d'] = pd.to_datetime(match_df['_d'])
        match_df = match_df.sort_values('_d').reset_index(drop=True)
        
        # 🎯 LogMatchStart의 타임스탬프를 0초 기준으로 사용
        logmatchstart_df = match_df[match_df['_t'] == 'logmatchstart']
        timestamp_elapsed0 = logmatchstart_df._d.iloc[0]
        
        match_df['ingame_time'] = (match_df._d - timestamp_elapsed0).apply(lambda x:x.total_seconds() if x.total_seconds() > 0 else 0)

        return match_df
    
    def remove_item_log(self, df):
        remove_subcate_list = ['Shirts', 'Mask', 'Gloves', 'Bottoms', 'Footwear', 'Eyewear', 'Jacket']
        df = df[~df.item_subcategory.isin(remove_subcate_list)]
        return df
    
    @time_checker
    def parse_pickup_item_seq(self, match_df) -> pd.DataFrame:
        # Row: accountId@matchId, ingame_time:[t, ], character_zone:[city, ], character_location:[(x,y,z), ], item_itemId:[item, ]
        # Source: LogItemPickup only
        
        # 🎯 LogItemPickup만 사용
        LogItemPickup_df = match_df[match_df._t == 'logitempickup']
        LogItemPickup_df = LogItemPickup_df.dropna(axis=1, how='all')
        
        # 픽업 로그가 없는 경우 빈 DataFrame 반환
        if len(LogItemPickup_df) == 0:
            print(f"  ⚠️ Warning: No pickup events found. Returning empty pickup stats.")
            return pd.DataFrame()
        
        LogItemPickup_df, new_cols = dict2cols(LogItemPickup_df, 'character')
        LogItemPickup_df, new_cols = dict2cols(LogItemPickup_df, 'item')
        
        # 필수 컬럼이 없는 경우 빈 DataFrame 반환
        required_cols = ['character_accountid', 'item_itemid']
        missing_cols = [col for col in required_cols if col not in LogItemPickup_df.columns]
        
        if missing_cols:
            print(f"  ⚠️ Warning: Missing required columns {missing_cols}. Returning empty pickup stats.")
            return pd.DataFrame()
        
        LogItemPickup_df = self.remove_item_log(LogItemPickup_df)
        
        # StartParachutePack 제외 (비행기에서 지급되는 초기 낙하산)
        # Item_Back_B_01_StartParachutePack_C는 게임 시작 시 비행기에서 자동으로 지급
        # BackupParachute(게임 중 획득 가능)는 유지
        is_start_parachute = LogItemPickup_df['item_itemid'].str.contains('StartParachutePack', na=False)
        LogItemPickup_df = LogItemPickup_df[~is_start_parachute]
        
        LogItemPickup_df['user_match'] = LogItemPickup_df['character_accountid'] + f'@{self.match_id}'
        
        user_pickup_df = LogItemPickup_df.groupby('user_match').agg(
            pickup_time = ('ingame_time', list),
            pickup_city = ('character_zone', list),
            pickup_coord = ('character_location', list),
            pickup_item = ('item_itemid', list),
            pickup_cate=('item_subcategory', list),
            pickup_stack = ('item_stackcount', list),  # 🎯 stackCount 추가
        )
        
        return user_pickup_df
    
    @time_checker
    def parse_attach_seq(self, match_df) -> pd.DataFrame: 
        
        LogItemAttach = match_df[match_df._t == 'logitemattach']
        LogItemAttach = LogItemAttach.dropna(axis=1, how='all')
        
        # 어태치 로그가 없는 경우 빈 DataFrame 반환
        if len(LogItemAttach) == 0:
            print(f"  ⚠️ Warning: No LogItemAttach events found. Returning empty attach stats.")
            return pd.DataFrame()
        
        LogItemAttach, item_cols = dict2cols(LogItemAttach, 'parentitem')
        LogItemAttach, item_cols = dict2cols(LogItemAttach, 'childitem')
        LogItemAttach, item_cols = dict2cols(LogItemAttach, 'character')
        
        # 필수 컬럼이 없는 경우 빈 DataFrame 반환
        required_cols = ['childitem_itemid', 'character_accountid', 'parentitem_subcategory']
        missing_cols = [col for col in required_cols if col not in LogItemAttach.columns]
        
        if missing_cols:
            print(f"  ⚠️ Warning: Missing required columns {missing_cols}. Returning empty attach stats.")
            return pd.DataFrame()
        
        LogItemAttach['childitem_cate'] = LogItemAttach['childitem_itemid'].apply(
            lambda x: x.split('_')[3] if x and len(x.split('_')) > 3 else None
        )

        LogItemAttach['user_match'] = LogItemAttach['character_accountid'] + f'@{self.match_id}' 
        LogItemAttach = LogItemAttach[(LogItemAttach.parentitem_subcategory == 'Main')]
        # print(LogItemAttach.parentitem_subcategory.value_counts())

        user_attach_df = LogItemAttach.groupby('user_match').agg(
            attach_time=('ingame_time', list),
            attach_parent=('parentitem_itemid', list),
            attach_child=('childitem_itemid', list),
            attach_child_cate=('childitem_cate', list)
        )
        
        return user_attach_df
    
    @time_checker
    def parse_drop_item_seq(self, match_df) -> pd.DataFrame: 
        # Row: accountId@matchId, ingame_time:[t, ], character_zone:[city, ], character_location:[(x,y,z), ], item_itemId:[item, ]
        # Source: LogItemDrop only
        
        # 🎯 LogItemDrop만 사용
        LogItemDrop = match_df[match_df._t == 'logitemdrop'].copy()
        LogItemDrop = LogItemDrop.dropna(axis=1, how='all')
        
        if len(LogItemDrop) == 0:
            print(f"  ⚠️ Warning: No LogItemDrop events found. Returning empty drop stats.")
            return pd.DataFrame()
        
        LogItemDrop, new_cols = dict2cols(LogItemDrop, 'character')
        LogItemDrop, new_cols = dict2cols(LogItemDrop, 'item')
        
        # 필수 컬럼이 없는 경우 빈 DataFrame 반환
        required_cols = ['character_accountid', 'ingame_time']
        missing_cols = [col for col in required_cols if col not in LogItemDrop.columns]
        
        if missing_cols:
            print(f"  ⚠️ Warning: Missing required columns {missing_cols}. Returning empty drop stats.")
            return pd.DataFrame()
        
        LogItemDrop = self.remove_item_log(LogItemDrop)
        LogItemDrop['user_match'] = LogItemDrop['character_accountid'] + f'@{self.match_id}'
        
        user_drop_df = LogItemDrop.groupby('user_match').agg(
            drop_time = ('ingame_time', list),
            drop_city = ('character_zone', list),
            drop_coord = ('character_location', list),
            drop_item = ('item_itemid', list),
            drop_stack = ('item_stackcount', list),  # 🎯 stackCount 추가
        )
        
        return user_drop_df
    
    @time_checker
    def parse_player_vehicle_seq(self, match_df):
        LogVehicleRide = match_df[match_df._t == 'logvehicleride']
        LogVehicleRide = LogVehicleRide.dropna(axis=1, how='all')
        LogVehicleRide['use_type'] = 'ride'

        LogVehicleLeave = match_df[match_df._t == 'logvehicleleave']
        LogVehicleLeave = LogVehicleLeave.dropna(axis=1, how='all')
        LogVehicleLeave['use_type'] = 'leave'
        
        LogVehicle = pd.concat([LogVehicleRide, LogVehicleLeave], axis=0).sort_values('_d')
        LogVehicle, new_cols = dict2cols(LogVehicle, 'character')
        LogVehicle, new_cols = dict2cols(LogVehicle, 'vehicle')
        
        # TransportAircraft (수송기) 제외 - 게임 시작 시 낙하산 비행기는 실제 차량이 아님
        LogVehicle = LogVehicle[~LogVehicle['vehicle_vehicleid'].str.contains('TransportAircraft', na=False)]
        
        LogVehicle['user_match'] = LogVehicle['character_accountid'] + f'@{self.match_id}'

        user_vehicle_df = LogVehicle.groupby('user_match').agg(
            vehicle_time=('ingame_time', list),
            vehicle_usetype=('use_type', list),
            vehicle_type=('vehicle_vehicleid', list)
        )
        
        return user_vehicle_df
    
    @time_checker
    def parse_aircraft_leave_seq(self, match_df):
        """
        TransportAircraft (수송기) 하차 시간 파싱
        - 착지 시작 시점을 명확히 하기 위해 사용
        """
        LogVehicleLeave = match_df[match_df._t == 'logvehicleleave']
        
        if LogVehicleLeave.empty:
            return pd.DataFrame()
        
        LogVehicleLeave = LogVehicleLeave.dropna(axis=1, how='all')
        LogVehicleLeave, new_cols = dict2cols(LogVehicleLeave, 'character')
        LogVehicleLeave, new_cols = dict2cols(LogVehicleLeave, 'vehicle')
        
        # TransportAircraft (수송기)만 필터링
        TransportAircraftLeave = LogVehicleLeave[
            LogVehicleLeave['vehicle_vehicleid'].str.contains('TransportAircraft', na=False)
        ].copy()  # .copy()로 SettingWithCopyWarning 방지
        
        if TransportAircraftLeave.empty:
            return pd.DataFrame()
        
        TransportAircraftLeave['user_match'] = TransportAircraftLeave['character_accountid'] + f'@{self.match_id}'
        
        # 각 플레이어의 첫 번째 수송기 하차 시간 (착지 시작 시점)
        aircraft_leave_df = TransportAircraftLeave.groupby('user_match').agg(
            aircraft_leave_time=('ingame_time', 'first')  # 첫 번째 하차 시간
        )
        
        return aircraft_leave_df
    
    @time_checker
    def parse_bluechip_revival_seq(self, match_df):
        """
        Bluechip Revival 파싱: LogVehicleRide에서 부활 비행기 탑승을 감지
        vehicleId가 "RedeployAircraft_DihorOtok_C" 또는 "RedeployAircraft_Tiger_C"인 경우
        """
        # Revival aircraft 목록
        REVIVAL_AIRCRAFT = [
            "RedeployAircraft_DihorOtok_C",
            "RedeployAircraft_Tiger_C"
        ]
        
        LogVehicleRide = match_df[match_df._t == 'logvehicleride']
        LogVehicleRide = LogVehicleRide.dropna(axis=1, how='all')
        
        # character, vehicle 정보 추출
        LogVehicleRide, new_cols = dict2cols(LogVehicleRide, 'character')
        LogVehicleRide, new_cols = dict2cols(LogVehicleRide, 'vehicle')
        
        # Revival aircraft만 필터링
        revival_logs = LogVehicleRide[LogVehicleRide['vehicle_vehicleid'].isin(REVIVAL_AIRCRAFT)].copy()
        
        if revival_logs.empty:
            # Revival이 없는 경우 빈 DataFrame 반환 (index 설정)
            return pd.DataFrame(columns=['revival_time', 'revival_aircraft']).rename_axis('user_match')
        
        revival_logs['user_match'] = revival_logs['character_accountid'] + f'@{self.match_id}'
        
        # 유저별 revival 이벤트 집계 (index로 user_match 설정)
        user_revival_df = revival_logs.groupby('user_match').agg(
            revival_time=('ingame_time', list),
            revival_aircraft=('vehicle_vehicleid', list)
        )
        
        return user_revival_df
    
    @time_checker
    def parse_player_info(self, match_df):
        mapname = match_df[match_df._t == 'logmatchstart'].iloc[0]['mapname']

        LogMatchStart = match_df[match_df._t == 'logmatchend']
        LogMatchStart = LogMatchStart.dropna(axis=1, how='all')
        LogMatchStart['mapname'] = mapname

        match_user_df = LogMatchStart[["characters", "common", "mapname"]].explode('characters', ignore_index=True)
        match_user_df["character"] = match_user_df["characters"].apply(lambda x: x["character"])
        characters_expanded = pd.json_normalize(match_user_df["characters"].apply(lambda x: {k: v for k, v in x.items() if k != "character"}))
        common_expanded = pd.json_normalize(match_user_df["common"])[["mapName"]]
        match_user_df = pd.concat([match_user_df["character"], characters_expanded,common_expanded, match_user_df['mapname']], axis=1)
        match_user_df["mapName"] = match_user_df.apply(
            lambda row: "Baltic_Main" if row["mapname"] == "erangel" or row["mapName"] == "Baltic_Main" else row["mapName"],
            axis=1
        )
        match_user_df = match_user_df.drop("mapname", axis=1)
        match_user_df, new_cols = dict2cols(match_user_df, 'character')
        match_user_df['user_match'] = match_user_df['character_accountid'].apply(lambda x: f'{x}@{self.match_id}')
        match_user_df = match_user_df.set_index('user_match')

        LogPlayerKillV2 = match_df[match_df._t == 'logplayerkillv2']
        LogPlayerKillV2 = LogPlayerKillV2.dropna(axis=1, how='all')
        LogPlayerKillV2, new_cols = dict2cols(LogPlayerKillV2, 'victim')
        LogPlayerKillV2['user_match'] = LogPlayerKillV2['victim_accountid'].apply(lambda x: f'{x}@{self.match_id}')
        LogPlayerKillV2 = (
            LogPlayerKillV2.sort_values('ingame_time')
            .drop_duplicates(subset='user_match', keep='first')  # 가장 첫 번째 죽음만 남김
            .set_index('user_match')
        )
        match_user_df['death_time'] = LogPlayerKillV2['ingame_time']  # NaN -> Winner
        match_user_df['ischeater'] = None
        match_user_df = match_user_df[['character_name', 'character_teamid','character_ranking', 'death_time', 'mapName']]
        
        return match_user_df
       
    @time_checker
    def parse_player_state_seq(self, match_df):
        
        def calculate_distance_ratio(x, y, center_x, center_y, r):
            """중심으로부터의 거리 / 반경 비율 계산"""
            distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
            return distance / r
        
        def is_alive_at_time(time, death_times, revival_times):
            """특정 시간에 유저가 살아있는지 확인 (death/revival 고려)"""
            if not death_times:
                return True
            
            # 현재 상태 추적: 처음에는 살아있음
            alive = True
            events = []
            for dt in death_times:
                events.append((dt, 'death'))
            for rt in revival_times:
                events.append((rt, 'revival'))
            events.sort(key=lambda x: x[0])
            
            for event_time, event_type in events:
                if event_time > time:
                    break
                if event_type == 'death':
                    alive = False
                elif event_type == 'revival':
                    alive = True
            
            return alive
        
        def calculate_zone_distances(row, zone_info, death_times_dict, revival_times_dict):
            """BlueZone과 WhiteZone 각각에 대해 거리 계산 (살아있는 유저만)"""
            state_time = np.array(row.state_time)
            character_location = np.array(row.character_location)
            
            # user_match에서 accountid 추출
            accountid = row.name.split('@')[0] if '@' in row.name else row.name
            death_times = death_times_dict.get(accountid, [])
            revival_times = revival_times_dict.get(accountid, [])
            
            dist_from_bluezone = -1 * np.ones_like(state_time, dtype=float)
            dist_from_whitezone = -1 * np.ones_like(state_time, dtype=float)
            
            if zone_info is None or zone_info.empty:
                return pd.Series({
                    'dist_from_bluezone': dist_from_bluezone,
                    'dist_from_whitezone': dist_from_whitezone
                })
            
            for t_idx, time in enumerate(state_time):
                # 🎯 살아있는 유저만 계산
                if not is_alive_at_time(time, death_times, revival_times):
                    continue
                
                # time 이전의 최신 zone 정보 찾기
                valid_zones = zone_info[zone_info['ingame_time'] <= time]
                if not valid_zones.empty:
                    latest_zone = valid_zones.iloc[-1]
                    loc = character_location[t_idx]
                    
                    # BlueZone (현재 자기장) 거리 계산
                    blue_center_x = latest_zone['bluezone_x']
                    blue_center_y = latest_zone['bluezone_y']
                    blue_radius = latest_zone['bluezone_radius']
                    
                    if blue_radius > 0:
                        dist_from_bluezone[t_idx] = calculate_distance_ratio(
                            loc['x'], loc['y'], blue_center_x, blue_center_y, blue_radius
                        )
                    
                    # WhiteZone (다음 안전지대) 거리 계산
                    white_center_x = latest_zone['whitezone_x']
                    white_center_y = latest_zone['whitezone_y']
                    white_radius = latest_zone['whitezone_radius']
                    
                    if white_radius > 0:
                        dist_from_whitezone[t_idx] = calculate_distance_ratio(
                            loc['x'], loc['y'], white_center_x, white_center_y, white_radius
                        )
            
            return pd.Series({
                'dist_from_bluezone': dist_from_bluezone,
                'dist_from_whitezone': dist_from_whitezone
            })
        
        # 🎯 Death time 정보 가져오기 (살아있는 유저만 계산하기 위해)
        LogPlayerKillV2 = match_df[match_df._t == 'logplayerkillv2']
        LogPlayerKillV2 = LogPlayerKillV2.dropna(axis=1, how='all')
        death_times_dict = {}
        if not LogPlayerKillV2.empty:
            LogPlayerKillV2, _ = dict2cols(LogPlayerKillV2, 'victim')
            for _, row in LogPlayerKillV2.iterrows():
                accountid = row.get('victim_accountid')
                if pd.notna(accountid):
                    if accountid not in death_times_dict:
                        death_times_dict[accountid] = []
                    death_times_dict[accountid].append(row['ingame_time'])
        
        # 🎯 Revival (Bluechip) 정보 가져오기
        revival_times_dict = {}
        revival_aircraft_logs = match_df[match_df._t == 'logrevivalaircraft']
        if not revival_aircraft_logs.empty:
            for _, row in revival_aircraft_logs.iterrows():
                revive_accounts = row.get('reviveaccounts', [])
                revival_time = row['ingame_time']
                if revive_accounts and isinstance(revive_accounts, list):
                    for account_id in revive_accounts:
                        if pd.notna(account_id):
                            if account_id not in revival_times_dict:
                                revival_times_dict[account_id] = []
                            revival_times_dict[account_id].append(revival_time)
        
        # 위치 데이터 수집: 모든 로그 타입 사용
        filtered_df = match_df
        
        # 🎯 위치 데이터 수집: character 필드가 있는 로그 + character가 없는 로그의 다른 필드들
        all_positions = []
        
        # 1. character 필드가 있는 로그
        char_logs = filtered_df[['ingame_time', 'character']].dropna(subset=['character'])
        if not char_logs.empty:
            char_logs, _ = dict2cols(char_logs, 'character')
            char_logs = char_logs[['ingame_time', 'character_location', 'character_accountid']]
            char_logs.columns = ['ingame_time', 'location', 'accountid']
            all_positions.append(char_logs)
        
        # 2. attacker 필드가 있는 로그 (LogPlayerTakeDamage, LogPlayerMakeGroggy, LogPlayerUseThrowable)
        attacker_logs = filtered_df[['ingame_time', 'attacker']].dropna(subset=['attacker'])
        if not attacker_logs.empty:
            attacker_logs, _ = dict2cols(attacker_logs, 'attacker')
            if 'attacker_location' in attacker_logs.columns and 'attacker_accountid' in attacker_logs.columns:
                attacker_logs = attacker_logs[['ingame_time', 'attacker_location', 'attacker_accountid']]
                attacker_logs.columns = ['ingame_time', 'location', 'accountid']
                all_positions.append(attacker_logs)
        
        # 3. victim 필드가 있는 로그 (LogPlayerTakeDamage, LogPlayerMakeGroggy, LogPlayerRevive)
        # 🎯 LogPlayerKillV2는 제외 - victim은 이미 죽었으므로 위치 업데이트 불필요
        victim_logs = filtered_df[filtered_df._t != 'logplayerkillv2'][['ingame_time', 'victim']].dropna(subset=['victim'])
        if not victim_logs.empty:
            victim_logs, _ = dict2cols(victim_logs, 'victim')
            if 'victim_location' in victim_logs.columns and 'victim_accountid' in victim_logs.columns:
                victim_logs = victim_logs[['ingame_time', 'victim_location', 'victim_accountid']]
                victim_logs.columns = ['ingame_time', 'location', 'accountid']
                all_positions.append(victim_logs)
        
        # 4. dbnomaker 필드가 있는 로그 (LogPlayerKillV2 - 기절시킨 사람)
        dbnomaker_logs = filtered_df[['ingame_time', 'dbnomaker']].dropna(subset=['dbnomaker'])
        if not dbnomaker_logs.empty:
            dbnomaker_logs, _ = dict2cols(dbnomaker_logs, 'dbnomaker')
            if 'dbnomaker_location' in dbnomaker_logs.columns and 'dbnomaker_accountid' in dbnomaker_logs.columns:
                dbnomaker_logs = dbnomaker_logs[['ingame_time', 'dbnomaker_location', 'dbnomaker_accountid']]
                dbnomaker_logs.columns = ['ingame_time', 'location', 'accountid']
                all_positions.append(dbnomaker_logs)
        
        # 5. finisher 필드가 있는 로그 (LogPlayerKillV2 - 처치한 사람)
        finisher_logs = filtered_df[['ingame_time', 'finisher']].dropna(subset=['finisher'])
        if not finisher_logs.empty:
            finisher_logs, _ = dict2cols(finisher_logs, 'finisher')
            if 'finisher_location' in finisher_logs.columns and 'finisher_accountid' in finisher_logs.columns:
                finisher_logs = finisher_logs[['ingame_time', 'finisher_location', 'finisher_accountid']]
                finisher_logs.columns = ['ingame_time', 'location', 'accountid']
                all_positions.append(finisher_logs)
        
        # 6. reviver 필드가 있는 로그 (LogPlayerRevive)
        # 🎯 reviver 컬럼이 없는 경우 처리 (짧은 경기에서 부활 이벤트가 없을 수 있음)
        if 'reviver' in filtered_df.columns:
            reviver_logs = filtered_df[['ingame_time', 'reviver']].dropna(subset=['reviver'])
        else:
            reviver_logs = pd.DataFrame()
        if not reviver_logs.empty:
            reviver_logs, _ = dict2cols(reviver_logs, 'reviver')
            if 'reviver_location' in reviver_logs.columns and 'reviver_accountid' in reviver_logs.columns:
                reviver_logs = reviver_logs[['ingame_time', 'reviver_location', 'reviver_accountid']]
                reviver_logs.columns = ['ingame_time', 'location', 'accountid']
                all_positions.append(reviver_logs)
        
        # 모든 위치 데이터 합치기
        if all_positions:
            PlayerPosition = pd.concat(all_positions, ignore_index=True)
            PlayerPosition = PlayerPosition.sort_values('ingame_time')  # 🎯 시간순 정렬
        else:
            PlayerPosition = pd.DataFrame(columns=['ingame_time', 'location', 'accountid'])
        
        PlayerPosition['user_match'] = PlayerPosition['accountid'] + f'@{self.match_id}'
        
        periodic_player_states = PlayerPosition.groupby('user_match').agg(
            state_time = ('ingame_time', list),
            character_location = ('location', list),
        )
        
        # BlueZone과 WhiteZone 거리를 각각 계산 (살아있는 유저만)
        zone_distances = periodic_player_states.apply(
            lambda row: calculate_zone_distances(row, self.zone_info, death_times_dict, revival_times_dict), axis=1
        )
        periodic_player_states['dist_from_bluezone'] = zone_distances['dist_from_bluezone']
        periodic_player_states['dist_from_whitezone'] = zone_distances['dist_from_whitezone']
        
        return periodic_player_states
    
    @time_checker
    def parse_num_near_players_seq_vectorized(self, match_df, d_meter=150, t=10) -> pd.DataFrame:
        
        def forward_fill_positions(t_list, xs):
            """
            가장 최근 LogPlayerPosition 값을 사용 (forward-fill)
            각 10초 간격 시간 포인트에서 그 시간 이전의 가장 최근 좌표를 사용
            """
            max_time = 2000
            t_new = list(range(0, min(int(max(t_list)) + 10, max_time), 10))
            
            # xs: [coord_x, coord_y, coord_z] 각각 리스트
            x_new = []
            for x_vals in xs:
                filled_vals = []
                for target_t in t_new:
                    # target_t 이하인 시간들 중 가장 최근 값 찾기
                    valid_indices = [i for i, orig_t in enumerate(t_list) if orig_t <= target_t]
                    if valid_indices:
                        latest_idx = max(valid_indices, key=lambda i: t_list[i])
                        filled_vals.append(x_vals[latest_idx])
                    else:
                        # target_t 이전에 데이터가 없으면 첫 번째 값 사용
                        filled_vals.append(x_vals[0] if x_vals else 0)
                x_new.append(filled_vals)
            
            x_new = np.array(x_new)
            C, T = x_new.shape
            t_new_arr = np.array(t_new)[np.newaxis, :]
            
            mask_t = np.array(range(0, max_time, 10))[np.newaxis,:]
            mask_x = np.zeros((C+1, max_time//10))
            mask_x[:-1, :T] = x_new
            mask_x[-1, :T] = 1
            return np.concatenate([mask_t, mask_x], axis=0)
        
        def compute_distance_matrix(positions, d):
            N, T, _ = positions.shape
            diff = positions[:, None, :, :] - positions[None, :, :, :]  # 사람들 간 차이 벡터 (N, N, T, 3)
            dist = np.linalg.norm(diff, axis=-1)  # (N, N, T)
            within_distance = (dist <= d).astype(int)  # (N, N, T)
            return within_distance[..., np.newaxis]
        
        LogPlayerCreate = match_df[match_df._t == 'logplayercreate']
        LogPlayerCreate = LogPlayerCreate.dropna(axis=1, how='all')
        LogPlayerCreate, _ = dict2cols(LogPlayerCreate, 'character')
        LogPlayerCreate = LogPlayerCreate.drop_duplicates('character_accountid')
        LogPlayerCreate = LogPlayerCreate.set_index(keys='character_accountid')
        
        # 🎯 Death time 정보 가져오기 (살아있는 유저만 카운트하기 위해)
        LogPlayerKillV2 = match_df[match_df._t == 'logplayerkillv2']
        LogPlayerKillV2 = LogPlayerKillV2.dropna(axis=1, how='all')
        LogPlayerKillV2, _ = dict2cols(LogPlayerKillV2, 'victim')
        
        # 각 유저의 모든 death time 리스트 (여러 번 죽을 수 있음)
        death_times_dict = {}
        for _, row in LogPlayerKillV2.iterrows():
            accountid = row.get('victim_accountid')
            if pd.notna(accountid):
                if accountid not in death_times_dict:
                    death_times_dict[accountid] = []
                death_times_dict[accountid].append(row['ingame_time'])
        
        # 🎯 Revival (Bluechip) 정보 가져오기
        revival_times_dict = {}
        revival_aircraft_logs = match_df[match_df._t == 'logrevivalaircraft']
        if not revival_aircraft_logs.empty:
            for _, row in revival_aircraft_logs.iterrows():
                revive_accounts = row.get('reviveaccounts', [])
                revival_time = row['ingame_time']
                if revive_accounts and isinstance(revive_accounts, list):
                    for account_id in revive_accounts:
                        if pd.notna(account_id):
                            if account_id not in revival_times_dict:
                                revival_times_dict[account_id] = []
                            revival_times_dict[account_id].append(revival_time)
        
        # 위치 데이터 수집: 모든 로그 타입 사용
        # 🎯 character 필드 + attacker/victim/killer/reviver 필드에서 위치 정보 수집
        filtered_df = match_df
        
        all_positions = []
        
        # 1. character 필드가 있는 로그
        char_logs = filtered_df[['ingame_time', 'character']].dropna(subset=['character'])
        if not char_logs.empty:
            char_logs, _ = dict2cols(char_logs, 'character')
            char_logs = char_logs[char_logs.character_accountid.apply(len) > 10]
            if not char_logs.empty:
                char_logs = char_logs[['ingame_time', 'character_name', 'character_location', 'character_accountid']]
                char_logs.columns = ['ingame_time', 'name', 'location', 'accountid']
                all_positions.append(char_logs)
        
        # 2. attacker 필드가 있는 로그
        attacker_logs = filtered_df[['ingame_time', 'attacker']].dropna(subset=['attacker'])
        if not attacker_logs.empty:
            attacker_logs, _ = dict2cols(attacker_logs, 'attacker')
            if all(col in attacker_logs.columns for col in ['attacker_name', 'attacker_location', 'attacker_accountid']):
                attacker_logs = attacker_logs[attacker_logs.attacker_accountid.apply(len) > 10]
                if not attacker_logs.empty:
                    attacker_logs = attacker_logs[['ingame_time', 'attacker_name', 'attacker_location', 'attacker_accountid']]
                    attacker_logs.columns = ['ingame_time', 'name', 'location', 'accountid']
                    all_positions.append(attacker_logs)
        
        # 3. victim 필드가 있는 로그
        # 🎯 LogPlayerKillV2는 제외 - victim은 이미 죽었으므로 위치 업데이트 불필요
        victim_logs = filtered_df[filtered_df._t != 'logplayerkillv2'][['ingame_time', 'victim']].dropna(subset=['victim'])
        if not victim_logs.empty:
            victim_logs, _ = dict2cols(victim_logs, 'victim')
            if all(col in victim_logs.columns for col in ['victim_name', 'victim_location', 'victim_accountid']):
                victim_logs = victim_logs[victim_logs.victim_accountid.apply(len) > 10]
                if not victim_logs.empty:
                    victim_logs = victim_logs[['ingame_time', 'victim_name', 'victim_location', 'victim_accountid']]
                    victim_logs.columns = ['ingame_time', 'name', 'location', 'accountid']
                    all_positions.append(victim_logs)
        
        # 4. dbnomaker 필드가 있는 로그 (LogPlayerKillV2 - 기절시킨 사람)
        dbnomaker_logs = filtered_df[['ingame_time', 'dbnomaker']].dropna(subset=['dbnomaker'])
        if not dbnomaker_logs.empty:
            dbnomaker_logs, _ = dict2cols(dbnomaker_logs, 'dbnomaker')
            if all(col in dbnomaker_logs.columns for col in ['dbnomaker_name', 'dbnomaker_location', 'dbnomaker_accountid']):
                dbnomaker_logs = dbnomaker_logs[dbnomaker_logs.dbnomaker_accountid.apply(len) > 10]
                if not dbnomaker_logs.empty:
                    dbnomaker_logs = dbnomaker_logs[['ingame_time', 'dbnomaker_name', 'dbnomaker_location', 'dbnomaker_accountid']]
                    dbnomaker_logs.columns = ['ingame_time', 'name', 'location', 'accountid']
                    all_positions.append(dbnomaker_logs)
        
        # 5. finisher 필드가 있는 로그 (LogPlayerKillV2 - 처치한 사람)
        finisher_logs = filtered_df[['ingame_time', 'finisher']].dropna(subset=['finisher'])
        if not finisher_logs.empty:
            finisher_logs, _ = dict2cols(finisher_logs, 'finisher')
            if all(col in finisher_logs.columns for col in ['finisher_name', 'finisher_location', 'finisher_accountid']):
                finisher_logs = finisher_logs[finisher_logs.finisher_accountid.apply(len) > 10]
                if not finisher_logs.empty:
                    finisher_logs = finisher_logs[['ingame_time', 'finisher_name', 'finisher_location', 'finisher_accountid']]
                    finisher_logs.columns = ['ingame_time', 'name', 'location', 'accountid']
                    all_positions.append(finisher_logs)
        
        # 6. reviver 필드가 있는 로그
        # 🎯 reviver 컬럼이 없는 경우 처리 (짧은 경기에서 부활 이벤트가 없을 수 있음)
        if 'reviver' in filtered_df.columns:
            reviver_logs = filtered_df[['ingame_time', 'reviver']].dropna(subset=['reviver'])
        else:
            reviver_logs = pd.DataFrame()
        if not reviver_logs.empty:
            reviver_logs, _ = dict2cols(reviver_logs, 'reviver')
            if all(col in reviver_logs.columns for col in ['reviver_name', 'reviver_location', 'reviver_accountid']):
                reviver_logs = reviver_logs[reviver_logs.reviver_accountid.apply(len) > 10]
                if not reviver_logs.empty:
                    reviver_logs = reviver_logs[['ingame_time', 'reviver_name', 'reviver_location', 'reviver_accountid']]
                    reviver_logs.columns = ['ingame_time', 'name', 'location', 'accountid']
                    all_positions.append(reviver_logs)
        
        # 모든 위치 데이터 합치기
        if all_positions:
            PlayerPosition = pd.concat(all_positions, ignore_index=True)
            PlayerPosition = PlayerPosition.sort_values('ingame_time')  # 🎯 시간순 정렬
        else:
            return pd.DataFrame()
        
        PlayerPosition['x'] = PlayerPosition.location.apply(lambda x: x['x'] if isinstance(x, dict) else 0)
        PlayerPosition['y'] = PlayerPosition.location.apply(lambda x: x['y'] if isinstance(x, dict) else 0)
        PlayerPosition['z'] = PlayerPosition.location.apply(lambda x: x['z'] if isinstance(x, dict) else 0)

        coord_df = PlayerPosition[PlayerPosition['ingame_time'] > 0].groupby('accountid').agg(
            character_name = ('name', 'first'),
            t = ('ingame_time', list),
            coord_x = ('x', list),
            coord_y = ('y', list),
            coord_z = ('z', list)
        )

        coord_df['txyz'] = coord_df.apply(lambda row: forward_fill_positions(row.t, [row.coord_x, row.coord_y, row.coord_z]).astype(int), axis=1)
        
        # 🎯 LogMatchStart의 character 순서를 사용하여 정렬
        # character_order: {teamId: [accountId1, accountId2, ...]}
        # 전체 순서 리스트 생성 (teamId 오름차순, 각 팀 내에서는 LogMatchStart 순서)
        character_order = getattr(self, 'character_order', {})
        ordered_account_ids = []
        
        if character_order:
            for team_id in sorted(character_order.keys()):
                for account_id in character_order[team_id]:
                    if account_id in coord_df.index:
                        ordered_account_ids.append(account_id)
        
        # coord_df에 있지만 character_order에 없는 경우 (예외 처리 또는 fallback)
        for account_id in coord_df.index:
            if account_id not in ordered_account_ids:
                ordered_account_ids.append(account_id)
        
        # 정렬된 순서로 team_ids 생성
        team_ids = LogPlayerCreate['character_teamid'].loc[ordered_account_ids]
        
        all_user_traj = np.stack(coord_df['txyz'].loc[team_ids.index].tolist(), axis=0)
        all_user_traj= all_user_traj.transpose(0,2,1)
        
        # 🎯 각 시간 포인트에서 플레이어들의 생존 상태 마스크 생성
        time_points = all_user_traj[0, :, 0]  # 시간 포인트 배열
        N = all_user_traj.shape[0]  # 플레이어 수
        T = len(time_points)  # 시간 포인트 개수
        
        # 각 플레이어의 생존 상태 마스크 (N, T)
        alive_mask = np.ones((N, T), dtype=int)
        for i, accountid in enumerate(team_ids.index):
            # 죽음과 부활 이벤트를 시간 순으로 처리
            death_times = death_times_dict.get(accountid, [])
            revival_times = revival_times_dict.get(accountid, [])
            
            # 모든 이벤트를 시간 순으로 정렬 (죽음: 0, 부활: 1)
            events = []
            for dt in death_times:
                events.append((dt, 0))  # 0 = 죽음
            for rt in revival_times:
                events.append((rt, 1))  # 1 = 부활
            events.sort()  # 시간 순으로 정렬
            
            # 각 이벤트를 alive_mask에 반영
            for event_time, event_type in events:
                if event_type == 0:  # 죽음
                    alive_mask[i, time_points >= event_time] = 0
                else:  # 부활
                    alive_mask[i, time_points >= event_time] = 1
        
        # 카운트할 때 상대방도 살아있는지 확인하기 위해 (N, N, T, 1) 형태로 변환
        alive_mask_self = alive_mask[:, None, :, None]  # 자신의 생존 상태 (N, 1, T, 1)
        alive_mask_other = alive_mask[None, :, :, None]  # 상대방의 생존 상태 (1, N, T, 1)
        
        dist_mat = compute_distance_matrix(all_user_traj[...,:-1], d_meter*100)
        valid_time_mat = all_user_traj[:, None, : ,-1:]    
        team_mat = (team_ids.values[:, None] == team_ids.values).astype(int)
        
        # 🎯 팀원/적 카운트 시 자신과 상대방 모두 살아있는 경우만 카운트
        team_dist_mat = dist_mat * team_mat[:, :, None,None] * valid_time_mat * alive_mask_self * alive_mask_other
        n_near_teamates_seq = team_dist_mat[:, :, :, 0].sum(axis=1)
        
        enemy_dist_mat = dist_mat * ~(team_mat[:, :, None,None].astype(bool)) * valid_time_mat * alive_mask_self * alive_mask_other
        n_near_enemies_seq = enemy_dist_mat[:, :, :, 0].sum(axis=1)

        num_near_players = pd.DataFrame(columns=['count_time', 'interpolated_coord', 'num_near_teammates', 'num_near_enemies'])
       
        num_near_players['interpolated_coord'] = coord_df.loc[team_ids.index]['txyz'].apply(lambda x:x.tolist()).tolist()
        num_near_players['num_near_teammates'] = n_near_teamates_seq.tolist()
        num_near_players['num_near_enemies'] = n_near_enemies_seq.tolist()
        
        num_near_players['count_time'] = None
        num_near_players['count_time'] = num_near_players['count_time'].apply(lambda x: all_user_traj[0, :, 0] if x is None else None)
        
        num_near_players['user_match'] = team_ids.index.tolist()
        num_near_players['user_match'] = num_near_players['user_match'].apply(lambda x: f'{x}@{self.match_id}')
        num_near_players['character_name2'] = coord_df.loc[team_ids.index]['character_name'].tolist()
        num_near_players = num_near_players.set_index('user_match')
        
        return num_near_players
        
        
    @time_checker
    def parse_user_kill_stat(self, match_df):
        LogPlayerKillV2 = match_df[match_df._t == 'logplayerkillv2']
        LogPlayerKillV2 = LogPlayerKillV2.dropna(axis=1, how='all')
        
        # 킬 로그가 없는 경우 빈 DataFrame 반환
        if len(LogPlayerKillV2) == 0:
            print(f"  ⚠️ Warning: No LogPlayerKillV2 events found. Returning empty kill stats.")
            return pd.DataFrame()
        
        LogPlayerKillV2, finisher_cols = dict2cols(LogPlayerKillV2, 'finisher')
        LogPlayerKillV2, dBNOMaker_cols = dict2cols(LogPlayerKillV2, 'dbnomaker')
        LogPlayerKillV2, dBNOMaker_info_cols = dict2cols(LogPlayerKillV2, 'dbnodamageinfo')
        LogPlayerKillV2, killer_cols = dict2cols(LogPlayerKillV2, 'killer')
        LogPlayerKillV2, victim_cols = dict2cols(LogPlayerKillV2, 'victim')

        # 필수 컬럼이 없는 경우 빈 DataFrame 반환
        required_cols = ['dbnomaker_accountid', 'finisher_accountid', 'killer_accountid', 
                        'dbnomaker_name', 'finisher_name', 'killer_name', 'victim_name']
        missing_cols = [col for col in required_cols if col not in LogPlayerKillV2.columns]
        
        if missing_cols:
            print(f"  ⚠️ Warning: Missing required columns {missing_cols}. Returning empty kill stats.")
            return pd.DataFrame()

        LogPlayerKillV2['dbno_user_match'] = LogPlayerKillV2['dbnomaker_accountid'] + '@' + self.match_id
        LogPlayerKillV2['finisher_user_match'] = LogPlayerKillV2['finisher_accountid'] + '@' + self.match_id
        LogPlayerKillV2['killer_user_match'] = LogPlayerKillV2['killer_accountid'] + '@' + self.match_id

        if (LogPlayerKillV2.dbnomaker_name == '').sum() > LogPlayerKillV2.dbnomaker_name.isnull().sum():
            dbno_none_idx = LogPlayerKillV2.dbnomaker_name == '' # table -> json 
        else:
            dbno_none_idx = LogPlayerKillV2.dbnomaker_name.isnull() 
        finish_kill_idx = LogPlayerKillV2['finisher_name'] == LogPlayerKillV2['killer_name']
        
        squad_last_kill_df = LogPlayerKillV2[dbno_none_idx & finish_kill_idx].groupby('killer_user_match').agg(
            squad_last_member_n_kill = ('killer_name', 'count'),
            squad_last_member_killer = ('killer_name', 'first'),
            squad_last_member_kill_time = ('ingame_time', list),
            squad_last_member_victims = ('victim_name', list),
        )
        
        dbno_seq_df = LogPlayerKillV2.groupby('dbno_user_match').agg(
            groggy_n_dnbo = ('dbnomaker_name', 'count'),
            groggy_maker = ('dbnomaker_name', 'first'),
            groggy_time = ('ingame_time', list),
            groggy_victims = ('victim_name', list),
        )
        
        finish_seq_df = LogPlayerKillV2.groupby('finisher_user_match').agg(
            finish_n_dnbo = ('finisher_name', 'count'),
            finisher = ('finisher_name', 'first'),
            finish_time = ('ingame_time', list),
            finish_victims = ('victim_name', list),
        )
        
        killer_seq_df = LogPlayerKillV2.groupby('killer_user_match').agg(
            kill_n_dnbo = ('killer_name', 'count'),
            killer = ('killer_name', 'first'),
            kill_time = ('ingame_time', list),
            kill_victims = ('victim_name', list),
        )

        killv2_stat_df = pd.concat([squad_last_kill_df, dbno_seq_df, finish_seq_df, killer_seq_df], axis=1)
        killv2_stat_df.columns = [f'killv2_{col}' for col in killv2_stat_df.columns]
        killv2_stat_df.index.name = 'user_match'
        return killv2_stat_df
    
    @time_checker
    def parse_takedamage_seq(self, match_df):
        LogPlayerTakeDamage = match_df[match_df._t == 'logplayertakedamage']
        LogPlayerTakeDamage = LogPlayerTakeDamage.dropna(axis=1, how='all')
        
        # 데미지 로그가 없는 경우 빈 DataFrame 반환
        if len(LogPlayerTakeDamage) == 0:
            print(f"  ⚠️ Warning: No LogPlayerTakeDamage events found. Returning empty damage stats.")
            return pd.DataFrame(), pd.DataFrame()
        
        LogPlayerTakeDamage, attacker_cols = dict2cols(LogPlayerTakeDamage, 'attacker')
        LogPlayerTakeDamage, victim_cols = dict2cols(LogPlayerTakeDamage, 'victim')
        
        # 필수 컬럼이 없는 경우 빈 DataFrame 반환
        required_cols = ['attacker_accountid', 'victim_accountid', 'attacker_location', 'victim_location']
        missing_cols = [col for col in required_cols if col not in LogPlayerTakeDamage.columns]
        
        if missing_cols:
            print(f"  ⚠️ Warning: Missing required columns {missing_cols}. Returning empty damage stats.")
            return pd.DataFrame(), pd.DataFrame()

        LogPlayerTakeDamage['td_distance'] = LogPlayerTakeDamage.apply(lambda row: calculate_distance(row.attacker_location, row.victim_location) if notNone(row.attacker_location) else 0, axis=1)
        LogPlayerTakeDamage['td_distance'] = LogPlayerTakeDamage['td_distance'].astype(int)
        
        LogPlayerTakeDamage['attack_user_match'] = LogPlayerTakeDamage['attacker_accountid'] + '@' + self.match_id
        LogPlayerTakeDamage['victim_user_match'] = LogPlayerTakeDamage['victim_accountid'] + '@' + self.match_id
        
        # 🎯 attackid groupby 없이 바로 attacker/victim별로 집계 (self-damage 포함 모든 케이스 처리)
        td_attacker_df = LogPlayerTakeDamage.groupby('attack_user_match').agg(
            td_attack_count = ('ingame_time', 'count'),
            td_attack_ingame_st = ('ingame_time', list),
            td_attack_ingame_ed = ('ingame_time', list),
            td_attack_weapon = ('damagecausername', list),
            td_attack_victim = ('victim_name', list),
            td_attack_damage = ('damage', list),
            td_attack_distance = ('td_distance', list),
        ).sort_values('td_attack_count')
        
        td_victim_df = LogPlayerTakeDamage.groupby('victim_user_match').agg(
            td_victim_count = ('ingame_time', 'count'),
            td_victim_ingame_st = ('ingame_time', list),
            td_victim_ingame_ed = ('ingame_time', list),
            td_victim_weapon = ('damagecausername', list),
            td_victim_attacker = ('attacker_name', list),
            td_victim_damage = ('damage', list),
            td_victim_distance = ('td_distance', list),
        ).sort_values('td_victim_count')
        
        td_df = pd.concat([td_attacker_df, td_victim_df], axis=1)
        td_df.index.name = 'user_match'
    
        return td_df
    
    @time_checker
    def parse_dbno_seq(self, match_df):
        LogPlayerMakeGroggy = match_df[match_df._t == 'logplayermakegroggy']
        LogPlayerMakeGroggy = LogPlayerMakeGroggy.dropna(axis=1, how='all')
        LogPlayerMakeGroggy, attacker_cols = dict2cols(LogPlayerMakeGroggy, 'attacker')
        LogPlayerMakeGroggy, victim_cols = dict2cols(LogPlayerMakeGroggy, 'victim')
        
        LogPlayerMakeGroggy = LogPlayerMakeGroggy[LogPlayerMakeGroggy['attacker_accountid'] != '']
        
        LogPlayerMakeGroggy['attack_user_match'] = LogPlayerMakeGroggy['attacker_accountid'] + '@' + self.match_id
        LogPlayerMakeGroggy['victim_user_match'] = LogPlayerMakeGroggy['victim_accountid'] + '@' + self.match_id

        dbno_maker_df = LogPlayerMakeGroggy.groupby('attack_user_match').agg(
            dbno_maker_time = ('ingame_time', list),
            dbno_maker_victim_name = ('victim_name', list),
            dbno_maker_damageTypeCategory = ('damagetypecategory', list),
            dbno_maker_damageCauserName = ('damagecausername', list),
            dbno_maker_damageReason = ('damagereason', list),
            dbno_maker_distance = ('distance', list),
            dbno_maker_isInBlueZone = ('attacker_isinbluezone', list),
        )

        dbno_victim_df = LogPlayerMakeGroggy.groupby('victim_user_match').agg(
            dbno_victim_time = ('ingame_time', list),
            dbno_victim_attacker_name = ('attacker_name', list),
            dbno_victim_damageTypeCategory = ('damagetypecategory', list),
            dbno_victim_damageCauserName = ('damagecausername', list),
            dbno_victim_damageReason = ('damagereason', list),
            dbno_victim_distance = ('distance', list),
            dbno_victim_isInBlueZone = ('victim_isinbluezone', list),
        )
        dbno_df = pd.concat([dbno_maker_df, dbno_victim_df], axis=1)
        dbno_df.index.name = 'user_match'
        return dbno_df
        
    @time_checker
    def parse_test(self):
        self.user_info = self.parse_player_info(self.match_df)
        self.user_info.info()
        
        self.killv2_seq = self.parse_user_kill_stat(self.match_df)
        self.killv2_seq.info()
        
        print(self.killv2_seq[['killv2_squad_last_member_killer', 'killv2_squad_last_member_kill_time', 'killv2_squad_last_member_victims']].head())
        
        self.td_seq = self.parse_takedamage_seq(self.match_df)
        self.td_seq.info()
        # groggy 일때 damage가 0으로 찍힘
        print(self.td_seq[['td_attack_count', 'td_attack_ingame_st', 'td_attack_victim', 'td_attack_damage', 'td_attack_distance']].head())
        
        self.dbno_df = self.parse_dbno_seq(self.match_df)
        self.dbno_df.info()
        
        print(self.dbno_df[['dbno_maker_time', 'dbno_maker_victim_name', 'dbno_maker_damageTypeCategory', 'dbno_maker_distance']].head())
        
        self.incircle_seq = self.parse_player_state_seq(self.match_df)
        self.incircle_seq.info()
        self.pickup_seq = self.parse_pickup_item_seq(self.match_df)
        self.pickup_seq.info()
        self.drop_seq = self.parse_drop_item_seq(self.match_df)
        self.drop_seq.info()
        self.attach_seq = self.parse_attach_seq(self.match_df)
        self.attach_seq.info()
        self.vehicle_seq = self.parse_player_vehicle_seq(self.match_df)
        self.vehicle_seq.info()
        self.attack_seq = self.parse_player_attack_seq(self.match_df)
        self.attack_seq.info()
                
        self.num_near_players_seq_vec = self.parse_num_near_players_seq_vectorized(self.match_df)
        self.num_near_players_seq_vec.info()
        
        testdf = self.dbno_df
        
        if len(testdf.reset_index().user_match.apply(lambda x:len(x.split('@')[0])).value_counts()) > 1:
            print(testdf.reset_index().user_match.apply(lambda x:len(x.split('@')[0])).value_counts())

            raise Exception()
    
    
    @time_checker
    def parse_health_status_seq(self, match_df) -> pd.DataFrame:
        """character 필드가 있는 모든 로그에서 체력 정보 추출 (parse_player_state_seq와 동일 방식)"""
        # 🎯 LogPlayerPosition만으로는 최신 체력을 놓칠 수 있음 (주기적 로그이므로)
        # character.health가 있는 모든 이벤트에서 체력 정보 수집
        
        # 체력 데이터 수집: 모든 로그 타입 사용
        filtered_df = match_df
        
        all_health_logs = []
        
        # 1. character 필드가 있는 로그
        char_logs = filtered_df[['ingame_time', 'character']].dropna(subset=['character'])
        if not char_logs.empty:
            char_logs = char_logs.copy()
            char_logs, _ = dict2cols(char_logs, 'character')
            if 'character_health' in char_logs.columns and 'character_accountid' in char_logs.columns:
                cols_to_use = ['ingame_time', 'character_health', 'character_accountid']
                if 'character_location' in char_logs.columns:
                    cols_to_use.append('character_location')
                if 'character_isinbluezone' in char_logs.columns:
                    cols_to_use.append('character_isinbluezone')
                char_logs = char_logs[cols_to_use]
                char_logs.columns = ['ingame_time', 'health', 'accountid'] + (['location'] if 'character_location' in cols_to_use else []) + (['in_bluezone'] if 'character_isinbluezone' in cols_to_use else [])
                all_health_logs.append(char_logs)
        
        # 2. attacker 필드가 있는 로그
        attacker_logs = filtered_df[['ingame_time', 'attacker']].dropna(subset=['attacker'])
        if not attacker_logs.empty:
            attacker_logs = attacker_logs.copy()
            attacker_logs, _ = dict2cols(attacker_logs, 'attacker')
            if 'attacker_health' in attacker_logs.columns and 'attacker_accountid' in attacker_logs.columns:
                cols_to_use = ['ingame_time', 'attacker_health', 'attacker_accountid']
                if 'attacker_location' in attacker_logs.columns:
                    cols_to_use.append('attacker_location')
                if 'attacker_isinbluezone' in attacker_logs.columns:
                    cols_to_use.append('attacker_isinbluezone')
                attacker_logs = attacker_logs[cols_to_use]
                attacker_logs.columns = ['ingame_time', 'health', 'accountid'] + (['location'] if 'attacker_location' in cols_to_use else []) + (['in_bluezone'] if 'attacker_isinbluezone' in cols_to_use else [])
                all_health_logs.append(attacker_logs)
        
        # 3. victim 필드가 있는 로그 (LogPlayerKillV2 제외 - victim은 이미 죽었으므로)
        victim_logs = filtered_df[filtered_df._t != 'logplayerkillv2'][['ingame_time', 'victim']].dropna(subset=['victim'])
        if not victim_logs.empty:
            victim_logs = victim_logs.copy()
            victim_logs, _ = dict2cols(victim_logs, 'victim')
            if 'victim_health' in victim_logs.columns and 'victim_accountid' in victim_logs.columns:
                cols_to_use = ['ingame_time', 'victim_health', 'victim_accountid']
                if 'victim_location' in victim_logs.columns:
                    cols_to_use.append('victim_location')
                if 'victim_isinbluezone' in victim_logs.columns:
                    cols_to_use.append('victim_isinbluezone')
                victim_logs = victim_logs[cols_to_use]
                victim_logs.columns = ['ingame_time', 'health', 'accountid'] + (['location'] if 'victim_location' in cols_to_use else []) + (['in_bluezone'] if 'victim_isinbluezone' in cols_to_use else [])
                all_health_logs.append(victim_logs)
        
        if not all_health_logs:
            return pd.DataFrame()
        
        # 모든 체력 로그 합치기
        health_logs = pd.concat(all_health_logs, ignore_index=True)
        
        # 누락된 컬럼 기본값 설정
        if 'location' not in health_logs.columns:
            health_logs['location'] = None
        if 'in_bluezone' not in health_logs.columns:
            health_logs['in_bluezone'] = None
        
        health_logs['user_match'] = health_logs['accountid'] + f'@{self.match_id}'
        health_logs = health_logs.sort_values('ingame_time')  # 🎯 시간순 정렬
        
        # 사용자별 체력 시계열 정보 집계
        user_health_df = health_logs.groupby('user_match').agg(
            health_time=('ingame_time', list),
            health_value=('health', list),
            health_location=('location', list),
            health_in_bluezone=('in_bluezone', list)
        )
        
        return user_health_df
    
    @time_checker  
    def parse_survival_status_seq(self, match_df) -> pd.DataFrame:
        """생존 상태 관련 로그들을 종합해서 생존 상태 추적 (죽음과 블루칩 부활만 추적)"""
        # Source: LogPlayerKillV2, LogRevivalAircraft
        # Note: 
        # - LogPlayerRevive는 groggy에서 일어나는 것이므로 여기서 제외
        # - LogPlayerMakeGroggy는 parse_dbno_seq에서 별도로 처리 (n_groggy feature용)
        # - Groggy는 살아있는 상태이므로 생존/사망 추적에 포함하지 않음
        
        survival_events = []
        
        # Kill 이벤트 처리 (완전히 죽음)
        kill_logs = match_df[match_df._t == 'logplayerkillv2']
        if not kill_logs.empty:
            kill_logs, _ = dict2cols(kill_logs, 'victim')
            for _, row in kill_logs.iterrows():
                if pd.notna(row.get('victim_accountid')):
                    survival_events.append({
                        'user_match': row['victim_accountid'] + f'@{self.match_id}',
                        'event_time': row['ingame_time'],
                        'event_type': 'KILLED',
                        'health': row.get('victim_health', 0)
                    })
        
        # 🎯 Revival Aircraft 이벤트 처리 (블루칩으로 죽었다가 부활)
        revival_aircraft_logs = match_df[match_df._t == 'logrevivalaircraft']
        if not revival_aircraft_logs.empty:
            for _, row in revival_aircraft_logs.iterrows():
                revive_accounts = row.get('reviveaccounts', [])
                if revive_accounts and isinstance(revive_accounts, list):
                    for account_id in revive_accounts:
                        if pd.notna(account_id):
                            survival_events.append({
                                'user_match': account_id + f'@{self.match_id}',
                                'event_time': row['ingame_time'],
                                'event_type': 'REVIVED',  # 블루칩 부활
                                'health': 100  # 블루칩으로 부활하면 체력 100
                            })
        
        if not survival_events:
            return pd.DataFrame()
        
        # DataFrame 생성 및 정렬
        survival_df = pd.DataFrame(survival_events)
        survival_df = survival_df.sort_values(['user_match', 'event_time'])
        
        # 사용자별 생존 이벤트 시계열 집계
        user_survival_df = survival_df.groupby('user_match').agg(
            survival_time=('event_time', list),
            survival_event=('event_type', list),
            survival_health=('health', list)
        )
        
        return user_survival_df
    
    @time_checker
    def parse_zone_info(self, match_df):
        """LogGameStatePeriodic에서 SafeZone(블루존)과 PoisonGasWarning(화이트존 - 다음 자기장) 정보 추출 (모든 스쿼드에 공통)"""
        # Source: LogGameStatePeriodic
        # Note: 이 정보는 모든 스쿼드에 공통이므로 한 번만 파싱
        
        periodic_logs = match_df[match_df._t == 'loggamestateperiodic']
        if periodic_logs.empty:
            return pd.DataFrame()
        
        periodic_logs, _ = dict2cols(periodic_logs, 'gamestate')
        periodic_logs, _ = dict2cols(periodic_logs, 'gamestate_safetyzoneposition')
        periodic_logs, _ = dict2cols(periodic_logs, 'gamestate_poisongaswarningposition')
        
        # SafeZone(블루존)과 PoisonGasWarning(화이트존 - 다음 자기장) 정보 추출
        zone_data = []
        for _, row in periodic_logs.iterrows():
            bluezone_x = row.get('gamestate_safetyzoneposition_x', 0)
            bluezone_y = row.get('gamestate_safetyzoneposition_y', 0)
            bluezone_radius = row.get('gamestate_safetyzoneradius', 0)
            
            whitezone_x = row.get('gamestate_poisongaswarningposition_x', 0)
            whitezone_y = row.get('gamestate_poisongaswarningposition_y', 0)
            whitezone_radius = row.get('gamestate_poisongaswarningradius', 0)
            
            # 🎯 WhiteZone이 (0, 0, 0)이면 BlueZone 기본값으로 대체
            # (초반에 다음 자기장이 아직 결정되지 않은 경우, 맵별 기본값 자동 적용)
            if whitezone_x == 0 and whitezone_y == 0 and whitezone_radius == 0:
                whitezone_x = bluezone_x
                whitezone_y = bluezone_y
                whitezone_radius = bluezone_radius
            
            zone_data.append({
                'ingame_time': row.get('ingame_time', 0),  # 🎯 elapsed_time 대신 ingame_time 사용 (다른 로그와 통일)
                'bluezone_x': bluezone_x,
                'bluezone_y': bluezone_y,
                'bluezone_radius': bluezone_radius,
                'whitezone_x': whitezone_x,
                'whitezone_y': whitezone_y,
                'whitezone_radius': whitezone_radius
            })
        
        if not zone_data:
            return pd.DataFrame()
        
        zone_df = pd.DataFrame(zone_data)
        zone_df = zone_df.sort_values('ingame_time').drop_duplicates()
        
        return zone_df
    
    @time_checker
    def parse_revival_from_groggy_seq(self, match_df) -> pd.DataFrame:
        """LogPlayerRevive에서 groggy 상태에서의 부활 정보 추출"""
        # Source: LogPlayerRevive
        # Returns: user별 groggy에서 부활한 시간 리스트
        
        revive_logs = match_df[match_df._t == 'logplayerrevive']
        if revive_logs.empty:
            return pd.DataFrame()
        
        revive_logs, _ = dict2cols(revive_logs, 'victim')
        
        revival_from_groggy_events = []
        for _, row in revive_logs.iterrows():
            if pd.notna(row.get('victim_accountid')):
                revival_from_groggy_events.append({
                    'user_match': row['victim_accountid'] + f'@{self.match_id}',
                    'revival_time': row['ingame_time']
                })
        
        if not revival_from_groggy_events:
            return pd.DataFrame()
        
        # DataFrame 생성 및 정렬
        revival_df = pd.DataFrame(revival_from_groggy_events)
        revival_df = revival_df.sort_values(['user_match', 'revival_time'])
        
        # 사용자별 groggy 부활 시간 집계
        user_revival_df = revival_df.groupby('user_match').agg(
            revival_from_groggy_time=('revival_time', list)
        )
        
        return user_revival_df
    
    @time_checker
    def parse_groggy_status_seq(self, match_df) -> pd.DataFrame:
        """Groggy 상태 추적: 기절 시작(LogPlayerMakeGroggy)과 종료(LogPlayerRevive, LogPlayerKillV2)"""
        # Source: LogPlayerMakeGroggy, LogPlayerRevive, LogPlayerKillV2
        # Returns: user별 groggy 상태 변화 시간 리스트
        
        groggy_events = []
        
        # Groggy 시작 이벤트 (기절)
        groggy_logs = match_df[match_df._t == 'logplayermakegroggy']
        if not groggy_logs.empty:
            groggy_logs, _ = dict2cols(groggy_logs, 'victim')
            for _, row in groggy_logs.iterrows():
                if pd.notna(row.get('victim_accountid')):
                    groggy_events.append({
                        'user_match': row['victim_accountid'] + f'@{self.match_id}',
                        'event_time': row['ingame_time'],
                        'event_type': 'GROGGY_START'
                    })
        
        # Groggy 종료 이벤트 (부활)
        revive_logs = match_df[match_df._t == 'logplayerrevive']
        if not revive_logs.empty:
            revive_logs, _ = dict2cols(revive_logs, 'victim')
            for _, row in revive_logs.iterrows():
                if pd.notna(row.get('victim_accountid')):
                    groggy_events.append({
                        'user_match': row['victim_accountid'] + f'@{self.match_id}',
                        'event_time': row['ingame_time'],
                        'event_type': 'GROGGY_END'
                    })
        
        # Groggy 중 죽음 (기절 상태에서 완전히 죽음)
        kill_logs = match_df[match_df._t == 'logplayerkillv2']
        if not kill_logs.empty:
            kill_logs, _ = dict2cols(kill_logs, 'victim')
            for _, row in kill_logs.iterrows():
                if pd.notna(row.get('victim_accountid')):
                    groggy_events.append({
                        'user_match': row['victim_accountid'] + f'@{self.match_id}',
                        'event_time': row['ingame_time'],
                        'event_type': 'GROGGY_END'  # 죽으면 groggy도 종료
                    })
        
        if not groggy_events:
            return pd.DataFrame()
        
        # DataFrame 생성 및 정렬
        groggy_df = pd.DataFrame(groggy_events)
        groggy_df = groggy_df.sort_values(['user_match', 'event_time'])
        
        # 사용자별 groggy 상태 시계열 집계
        user_groggy_df = groggy_df.groupby('user_match').agg(
            groggy_status_time=('event_time', list),
            groggy_status_event=('event_type', list)
        )
        
        return user_groggy_df
    
    @time_checker
    def parse_heal_seq(self, match_df) -> pd.DataFrame:
        """LogHeal에서 힐 사용 정보 추출"""
        # Source: LogHeal
        # Returns: user별 힐 사용 시간, 아이템, 회복량
        
        heal_logs = match_df[match_df._t == 'logheal']
        if heal_logs.empty:
            return pd.DataFrame()
        
        heal_logs = heal_logs.dropna(axis=1, how='all')
        heal_logs, _ = dict2cols(heal_logs, 'character')
        heal_logs, _ = dict2cols(heal_logs, 'item')
        heal_logs['user_match'] = heal_logs['character_accountid'] + f'@{self.match_id}'
        
        user_heal_df = heal_logs.groupby('user_match').agg(
            heal_time=('ingame_time', list),
            heal_item=('item_itemid', list),
            heal_amount=('healamount', list)
        )
        
        return user_heal_df
    
    @time_checker
    def parse_equip_item_seq(self, match_df) -> pd.DataFrame:
        """LogItemEquip에서 장착 아이템 정보 추출"""
        # Source: LogItemEquip
        # Returns: user별 장착 아이템 시간, 아이템 ID
        
        equip_logs = match_df[match_df._t == 'logitemequip']
        if equip_logs.empty:
            return pd.DataFrame()
        
        equip_logs = equip_logs.dropna(axis=1, how='all')
        equip_logs, _ = dict2cols(equip_logs, 'character')
        equip_logs, _ = dict2cols(equip_logs, 'item')
        equip_logs['user_match'] = equip_logs['character_accountid'] + f'@{self.match_id}'
        
        user_equip_df = equip_logs.groupby('user_match').agg(
            equip_time=('ingame_time', list),
            equip_item=('item_itemid', list)
        )
        
        return user_equip_df
    
    @time_checker
    def parse_use_item_seq(self, match_df) -> pd.DataFrame:
        """LogItemUse에서 아이템 사용 정보 추출"""
        # Source: LogItemUse
        # Returns: user별 아이템 사용 시간, 아이템 ID
        
        use_logs = match_df[match_df._t == 'logitemuse']
        if use_logs.empty:
            return pd.DataFrame()
        
        use_logs = use_logs.dropna(axis=1, how='all')
        use_logs, _ = dict2cols(use_logs, 'character')
        use_logs, _ = dict2cols(use_logs, 'item')
        use_logs['user_match'] = use_logs['character_accountid'] + f'@{self.match_id}'
        
        user_use_df = use_logs.groupby('user_match').agg(
            use_time=('ingame_time', list),
            use_item=('item_itemid', list)
        )
        
        return user_use_df
    
    @time_checker
    def parse_throwable_use_seq(self, match_df) -> pd.DataFrame:
        """LogPlayerUseThrowable에서 투척 아이템 사용 정보 추출"""
        # Source: LogPlayerUseThrowable
        # Returns: user별 투척 아이템 사용 시간, 아이템 ID
        
        throwable_logs = match_df[match_df._t == 'logplayerusethrowable']
        if throwable_logs.empty:
            return pd.DataFrame()
        
        throwable_logs = throwable_logs.dropna(axis=1, how='all')
        throwable_logs, _ = dict2cols(throwable_logs, 'attacker')
        throwable_logs, _ = dict2cols(throwable_logs, 'weapon')
        throwable_logs['user_match'] = throwable_logs['attacker_accountid'] + f'@{self.match_id}'
        
        user_throwable_use_df = throwable_logs.groupby('user_match').agg(
            throwable_use_time=('ingame_time', list),
            throwable_use_item=('weapon_itemid', list)
        )
        
        return user_throwable_use_df
        
    def parse(self):
        user_info = self.parse_player_info(self.match_df)
        incircle_seq = self.parse_player_state_seq(self.match_df)
        pickup_seq = self.parse_pickup_item_seq(self.match_df)
        drop_seq = self.parse_drop_item_seq(self.match_df)
        attach_seq = self.parse_attach_seq(self.match_df)
        vehicle_seq = self.parse_player_vehicle_seq(self.match_df)
        num_near_players_seq = self.parse_num_near_players_seq_vectorized(self.match_df)
        
        kill_seq = self.parse_user_kill_stat(self.match_df)
        dbno_seq = self.parse_dbno_seq(self.match_df)
        td_seq = self.parse_takedamage_seq(self.match_df)
        
        # 🎯 새로운 Feature 파싱
        health_seq = self.parse_health_status_seq(self.match_df)
        survival_seq = self.parse_survival_status_seq(self.match_df)
        revival_from_groggy_seq = self.parse_revival_from_groggy_seq(self.match_df)
        groggy_status_seq = self.parse_groggy_status_seq(self.match_df)
        
        # 🎯 새로운 Item/Vehicle Feature 파싱
        heal_seq = self.parse_heal_seq(self.match_df)
        equip_seq = self.parse_equip_item_seq(self.match_df)
        use_seq = self.parse_use_item_seq(self.match_df)
        throwable_use_seq = self.parse_throwable_use_seq(self.match_df)
        
        # 🎯 Bluechip Revival 파싱
        bluechip_revival_seq = self.parse_bluechip_revival_seq(self.match_df)
        
        # 🎯 TransportAircraft 하차 시간 파싱 (착지 시작 시점)
        aircraft_leave_seq = self.parse_aircraft_leave_seq(self.match_df)
        
        parsed_match_df = pd.concat(
            [
                user_info, pickup_seq, drop_seq, attach_seq,
                incircle_seq, vehicle_seq, num_near_players_seq,
                kill_seq, dbno_seq, td_seq,
                health_seq, survival_seq, revival_from_groggy_seq, groggy_status_seq,
                heal_seq, equip_seq, use_seq, throwable_use_seq,
                bluechip_revival_seq, aircraft_leave_seq,
            ],
            axis=1
        )
        
        if len(parsed_match_df.reset_index().user_match.apply(lambda x:len(x.split('@')[0])).value_counts()) > 1:
            parsed_match_df['user_match'] = parsed_match_df.index.tolist()
            account_len_series = parsed_match_df.user_match.apply(len)
            valid_index = parsed_match_df[account_len_series >= self.user_match_str_len].index
            parsed_match_df = parsed_match_df.loc[valid_index].drop('user_match', axis=1)
    
        return parsed_match_df
        
    
def processing_match_log(match_log_json_path):
    log_parser = MatchLogParser(match_log_json_path)
    parsed_match_df = log_parser.parse()

    return parsed_match_df
        
def parallel_parsing(match_json_list):
    all_user_match_df = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(processing_match_log, json_path): json_path for json_path in match_json_list}

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            json_path = futures[future]
            if i % 50 == 0:
                print(i, json_path)
            try:
                match_user_df = future.result()
                if match_user_df is not None:
                    all_user_match_df.append(match_user_df.copy())
                    
            except Exception as e:
                print('error processing', json_path, e)

    all_user_match_df = pd.concat(all_user_match_df)
    all_user_match_df = all_user_match_df.reset_index().drop_duplicates('user_match').set_index('user_match')
    all_user_match_df.info()
    all_user_match_df.head()
    
    return all_user_match_df
        
if __name__ == '__main__':
    match_log_json_path = 'data/ded-data/match_logs/match.bro.competitive.pc-2018-32.steam.squad.as.2024.11.26.02.8cbb0ced-353b-4bc4-8b08-d02b0574908d.json'

    print(match_log_json_path)
    log_parser = MatchLogParser(match_log_json_path, source='ded')
    log_parser.parse_test()