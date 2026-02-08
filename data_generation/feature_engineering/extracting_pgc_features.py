import os
import pandas as pd
import numpy as np
import math

print(os.getcwd())
import sys
sys.path.append(os.getcwd())

from feature_engineering.utils import most_common_element, notNone, save_dict, load_dict
from feature_engineering.attach_score import AR_WEAPONS, DMR_WEAPONS, SR_WEAPONS, SMG_WEAPONS, SHOTGUN_WEAPONS, THROWABLE_WEAPONS
from feature_engineering.variables import PHASES_START, PHASES_MOVE, PHASES_END, PHASES_TIME, FEATURES1_PER_PHASE, HELMET_ITEMS, VEST_ITEMS, BACKPACK_ITEMS, THROWABLE_ITEMS, AIDS_ITEMS


td_AR_WEAPONS = [a.split('_')[2] for a in AR_WEAPONS]
td_DMR_WEAPONS = [a.split('_')[2] for a in DMR_WEAPONS]
td_SR_WEAPONS = [a.split('_')[2] for a in SR_WEAPONS]
td_SMG_WEAPONS = [a.split('_')[2] for a in SMG_WEAPONS]
td_SHOTGUN_WEAPONS = [a.split('_')[2] for a in SHOTGUN_WEAPONS]
td_THROWABLE_WEAPONS = [a.split('_')[2] for a in THROWABLE_WEAPONS]

class FeatureExtractor:
    def __init__(self, all_user_match_df=None):
        self.all_user_match_df = all_user_match_df

class SquadFeatureExtractor(FeatureExtractor):
    def __init__(self, all_user_match_df=None, squad_info=None, zone_info=None):
        super().__init__(all_user_match_df)  # 부모 클래스 초기화
        self.squad_info = squad_info  # user -> squad mapping
        self.zone_info = zone_info  # 🎯 Zone 정보: BlueZone(현재 자기장), WhiteZone(다음 안전지대) (게임 공통)
        
        # Feature 분류 정의
        self.action_cumulative_features = [
            'n_pickup', 'n_drop', 'n_kill', 'n_finish', 'n_groggy', 
            'n_dmg', 'sum_dmg', 'n_take_dmg', 'sum_take_dmg',
            # 🎯 총기별 dmg features
            'n_dmg_by_AR', 'n_dmg_by_SR', 'n_dmg_by_DMR', 'n_dmg_by_SMG', 'n_dmg_by_SHOTGUN',
            'sum_dmg_by_AR', 'sum_dmg_by_SR', 'sum_dmg_by_DMR', 'sum_dmg_by_SMG', 'sum_dmg_by_SHOTGUN',
            # 🎯 Heal cumulative features
            'n_heal', 'sum_heal_amount',
            'n_heal_bandage', 'n_heal_firstaid', 'n_heal_medkit',
            'n_heal_painkiller', 'n_heal_energydrink', 'n_heal_adrenaline',
            # 🎯 Vehicle cumulative features
            'n_vehicle_ride'
        ]
        
        self.snapshot_features = [
            'n_teammates', 'n_enemies', 'dist_from_bluezone', 'dist_from_whitezone',
            # 🎯 V2: Squad center based zone distances는 squad-level에서 직접 계산 (user aggregation 제외)
            'current_health', 'is_alive', 'is_out_bluezone', 'is_groggy',
            # 🎯 Vehicle snapshot features
            'is_vehicle_ride',
            # 🎯 Armor/Backpack snapshot features
            'item_helmet_level', 'item_vest_level', 'item_backpack_level',
            # 🎯 Throwables snapshot features (보유 개수)
            'item_throwables_count', 'item_grenade_count', 'item_smoke_count',
            'item_flashbang_count', 'item_molotov_count', 'item_bluezone_grenade_count',
            # 🎯 Aids snapshot features (보유 개수)
            'item_aids_count', 'item_bandage_count', 'item_firstaid_count',
            'item_medkit_count', 'item_painkiller_count', 'item_energydrink_count', 'item_adrenaline_count'
        ]
        
        # Phase 설정 (각 phase별로 동일한 개수 샘플링)
        self.phase_config = {
            1: {'start': 0, 'end': 600, 'n_samples': 5},
            2: {'start': 600, 'end': 810, 'n_samples': 5},
            3: {'start': 810, 'end': 990, 'n_samples': 5},
            4: {'start': 990, 'end': 1170, 'n_samples': 5},
            5: {'start': 1170, 'end': 1350, 'n_samples': 5},
            6: {'start': 1350, 'end': 1530, 'n_samples': 5},
            7: {'start': 1530, 'end': 1680, 'n_samples': 5},
            8: {'start': 1680, 'end': 1800, 'n_samples': 5},
            9: {'start': 1800, 'end': 1970, 'n_samples': 5},
            10: {'start': 1970, 'end': 2000, 'n_samples': 5}
        }
    
    def get_character_name(self, parsed_df):
        return parsed_df['character_name']
    
    def _get_squad_positions_at_timepoint(self, time_point):
        """
        특정 time_point에서 squad 내 모든 유저의 위치 정보를 추출
        🎯 Revival 시스템을 반영하여 부활한 유저는 살아있는 것으로 처리
        🎯 LogMatchStart 순서대로 정렬 (match_start_order 사용)
        """
        squad_positions = {}
        
        for squad_num in sorted(self.squad_info['squad_number'].unique()):
            # 해당 squad의 유저들 (LogMatchStart 순서대로 정렬)
            squad_user_names = self.squad_info[self.squad_info['squad_number'] == squad_num]
            if 'match_start_order' in squad_user_names.columns:
                squad_user_names = squad_user_names.sort_values('match_start_order')
            else:
                squad_user_names = squad_user_names.sort_values('character_name')  # fallback
            squad_users = squad_user_names['character_name']
            
            positions = []
            for user_name in squad_users:
                # 해당 유저의 데이터 찾기
                user_data = self.all_user_match_df[self.all_user_match_df['character_name'] == user_name]
                
                if len(user_data) > 0:
                    user_data = user_data.iloc[0]
                    
                    # 🎯 생존 여부 확인 (revival 반영)
                    is_alive = self._check_alive_at_timepoint(user_data, time_point)
                    
                    # time_point에서 죽어있으면 (0, 0, 0)
                    if not is_alive:
                        positions.append([0.0, 0.0, 0.0])
                    else:
                        # time_point 이전의 위치 데이터 찾기
                        state_times = np.array(user_data['state_time'])
                        character_locations = user_data['character_location']
                        
                        # time_point 이전의 가장 최근 시간 위치 찾기
                        valid_mask = state_times <= time_point
                        if np.any(valid_mask):
                            # 🎯 valid한 시간들 중에서 최대값(가장 최근)의 인덱스 찾기
                            masked_times = np.where(valid_mask, state_times, -np.inf)
                            last_idx = np.argmax(masked_times)
                            location = character_locations[last_idx]
                            
                            x = location.get('x', 0.0)
                            y = location.get('y', 0.0) 
                            z = location.get('z', 0.0)
                            positions.append([x, y, z])
                        else:
                            # time_point 이전 데이터가 없으면 기본값
                            positions.append([0.0, 0.0, 0.0])
                else:
                    # 유저 데이터가 없으면 기본값
                    positions.append([0.0, 0.0, 0.0])
         
            while len(positions) < 4:
                positions.append([0.0, 0.0, 0.0])
                    
            squad_positions[squad_num] = positions
        
        return squad_positions
    
    def _check_alive_at_timepoint(self, user_data, time_point):
        """
        특정 time_point에서 유저가 살아있는지 확인 (bluechip revival 반영)
        🎯 survival_time (KILLED) + revival_time (REVIVED) 통합 사용
        Returns: True if alive, False if dead
        """
        # 🎯 survival_time에서 KILLED 이벤트 가져오기
        survival_times = user_data.get('survival_time', [])
        if survival_times is None or (isinstance(survival_times, float) and pd.isna(survival_times)):
            survival_times = []
        
        # 🎯 revival_time에서 REVIVED 이벤트 가져오기
        revival_times = user_data.get('revival_time', [])
        if revival_times is not None and not isinstance(revival_times, (list, np.ndarray, pd.Series)):
            revival_times = [revival_times] if not pd.isna(revival_times) else []
        if revival_times is None or (isinstance(revival_times, float) and pd.isna(revival_times)):
            revival_times = []
        
        # 생사 이벤트가 없으면 살아있음
        if not survival_times and not revival_times:
            return True
        
        # 🎯 두 이벤트 리스트 통합 후 시간순 정렬
        life_death_events = []
        for t in survival_times:
            life_death_events.append((t, 'KILLED'))
        for t in revival_times:
            life_death_events.append((t, 'REVIVED'))
        life_death_events.sort(key=lambda x: x[0])
        
        # time_point까지의 이벤트를 순차적으로 처리
        is_alive = True
        for event_time, event_type in life_death_events:
            if event_time <= time_point:
                if event_type == 'KILLED':
                    is_alive = False
                elif event_type == 'REVIVED':
                    is_alive = True
        
        return is_alive
        
    def get_squad_features_at_timepoint(self, time_point):
        """
        특정 time_point까지의 누적 feature를 squad 단위로 집계
        """
        
        # 0부터 time_point까지의 누적 feature 생성
        user_features = self.get_feature_set1(
            PHASES_CUSTOM=[0, time_point], 
            PHASES_IDX=1
        )
        
        # 🎯 Snapshot features: 특정 시점의 값만 계산
        user_features['n_teammates'] = self.feature_n_teammates_at_timepoint(self.all_user_match_df, time_point)
        user_features['n_enemies'] = self.feature_n_enemies_at_timepoint(self.all_user_match_df, time_point)
        user_features['dist_from_bluezone'] = self.feature_dist_from_bluezone_at_timepoint(self.all_user_match_df, time_point)
        user_features['dist_from_whitezone'] = self.feature_dist_from_whitezone_at_timepoint(self.all_user_match_df, time_point)
        # 🎯 V2는 squad-level에서 직접 계산 (이미 계산된 positions 활용)
        user_features['current_health'] = self.feature_current_health(self.all_user_match_df, time_point)
        user_features['is_alive'] = self.feature_is_alive(self.all_user_match_df, time_point)
        user_features['is_out_bluezone'] = self.feature_is_out_bluezone(self.all_user_match_df, time_point)
        user_features['is_groggy'] = self.feature_is_groggy(self.all_user_match_df, time_point)
        
        # 🎯 새로운 Heal Features (LogHeal 기반 - 체력 회복 아이템)
        user_features['n_heal'] = self.feature_n_heal(self.all_user_match_df, time_point)
        user_features['sum_heal_amount'] = self.feature_sum_heal_amount(self.all_user_match_df, time_point)
        user_features['n_heal_bandage'] = self.feature_n_heal_by_item(self.all_user_match_df, time_point, 'bandage')
        user_features['n_heal_firstaid'] = self.feature_n_heal_by_item(self.all_user_match_df, time_point, 'firstaid')
        user_features['n_heal_medkit'] = self.feature_n_heal_by_item(self.all_user_match_df, time_point, 'medkit')
        
        # 🎯 Boost Features (LogItemUse 기반)
        user_features['n_heal_painkiller'] = self.feature_n_boost_by_item(self.all_user_match_df, time_point, 'painkiller')
        user_features['n_heal_energydrink'] = self.feature_n_boost_by_item(self.all_user_match_df, time_point, 'energydrink')
        user_features['n_heal_adrenaline'] = self.feature_n_boost_by_item(self.all_user_match_df, time_point, 'adrenaline')
        
        # 🎯 새로운 Vehicle Features
        user_features['is_vehicle_ride'] = self.feature_vehicle_has(self.all_user_match_df, time_point)
        user_features['n_vehicle_ride'] = self.feature_n_vehicle_ride(self.all_user_match_df, time_point)
        
        # 🎯 새로운 Armor/Backpack Features
        user_features['item_helmet_level'] = self.feature_item_helmet_level(self.all_user_match_df, time_point)
        user_features['item_vest_level'] = self.feature_item_vest_level(self.all_user_match_df, time_point)
        user_features['item_backpack_level'] = self.feature_item_backpack_level(self.all_user_match_df, time_point)
        
        # 🎯 새로운 Throwables Features (보유 개수)
        user_features['item_throwables_count'] = self.feature_n_throwables(self.all_user_match_df, time_point, 'total')
        user_features['item_grenade_count'] = self.feature_n_throwables(self.all_user_match_df, time_point, 'grenade')
        user_features['item_smoke_count'] = self.feature_n_throwables(self.all_user_match_df, time_point, 'smoke')
        user_features['item_flashbang_count'] = self.feature_n_throwables(self.all_user_match_df, time_point, 'flashbang')
        user_features['item_molotov_count'] = self.feature_n_throwables(self.all_user_match_df, time_point, 'molotov')
        user_features['item_bluezone_grenade_count'] = self.feature_n_throwables(self.all_user_match_df, time_point, 'bluezone_grenade')
        
        # 🎯 새로운 Aids Features (보유 개수)
        user_features['item_aids_count'] = self.feature_n_aids(self.all_user_match_df, time_point, 'total')
        user_features['item_bandage_count'] = self.feature_n_aids(self.all_user_match_df, time_point, 'bandage')
        user_features['item_firstaid_count'] = self.feature_n_aids(self.all_user_match_df, time_point, 'firstaid')
        user_features['item_medkit_count'] = self.feature_n_aids(self.all_user_match_df, time_point, 'medkit')
        user_features['item_painkiller_count'] = self.feature_n_aids(self.all_user_match_df, time_point, 'painkiller')
        user_features['item_energydrink_count'] = self.feature_n_aids(self.all_user_match_df, time_point, 'energydrink')
        user_features['item_adrenaline_count'] = self.feature_n_aids(self.all_user_match_df, time_point, 'adrenaline')
        
        # Squad 정보와 결합
        if self.squad_info is not None:
            user_features = user_features.merge(
                self.squad_info[['character_name', 'squad_number']], 
                on='character_name', 
                how='left'
            )
        else:
            # Squad 정보가 없으면 임시로 character_name을 squad로 사용
            user_features['squad_number'] = user_features['character_name']
        
        # Squad별 feature 집계
        squad_features = self._aggregate_squad_features(user_features)

        # Squad별 위치 정보 추가
        squad_positions = self._get_squad_positions_at_timepoint(time_point)
        squad_features['positions'] = squad_features['squad_number'].map(squad_positions)
        
        # 🎯 Zone 정보 추가 (모든 squad에 공통)
        zone_data = self._get_zone_center_at_timepoint(time_point)
        
        # 🎯 V2: Squad 중심 기반 zone 거리 계산 (이미 계산된 positions 활용)
        squad_features = self._calculate_squad_zone_distances_v2(squad_features, zone_data)
        
        # BlueZone (현재 자기장) 정보
        bluezone_tuple = (
            float(zone_data['bluezone_center_x']), 
            float(zone_data['bluezone_center_y']), 
            float(zone_data['bluezone_radius'])
        )
        squad_features['bluezone_info'] = [bluezone_tuple] * len(squad_features)
        
        # WhiteZone (다음 안전지대) 정보
        whitezone_tuple = (
            float(zone_data['whitezone_center_x']), 
            float(zone_data['whitezone_center_y']), 
            float(zone_data['whitezone_radius'])
        )
        squad_features['whitezone_info'] = [whitezone_tuple] * len(squad_features)
        
        return squad_features
    
    def _get_zone_center_at_timepoint(self, time_point):
        """특정 시점에서 BlueZone(현재 자기장)과 WhiteZone(다음 안전지대) 정보 추출"""
        
        # Zone 정보가 없으면 기본값 반환
        if self.zone_info is None or self.zone_info.empty:
            return {
                'bluezone_center_x': 408000,  # 기본 맵 중심
                'bluezone_center_y': 408000,
                'bluezone_radius': 581999.125,
                'whitezone_center_x': 408000,
                'whitezone_center_y': 408000,
                'whitezone_radius': 581999.125
            }
        
        # time_point 이전의 최신 zone 정보 찾기
        valid_zones = self.zone_info[self.zone_info['ingame_time'] <= time_point]
        
        if not valid_zones.empty:
            # 가장 최신 정보 사용
            latest_zone = valid_zones.iloc[-1]
            return {
                'bluezone_center_x': latest_zone['bluezone_x'],
                'bluezone_center_y': latest_zone['bluezone_y'],
                'bluezone_radius': latest_zone['bluezone_radius'],
                'whitezone_center_x': latest_zone['whitezone_x'],
                'whitezone_center_y': latest_zone['whitezone_y'],
                'whitezone_radius': latest_zone['whitezone_radius']
            }
        else:
            # time_point 이전 데이터가 없으면 첫 번째 값 사용
            first_zone = self.zone_info.iloc[0]
            return {
                'bluezone_center_x': first_zone['bluezone_x'],
                'bluezone_center_y': first_zone['bluezone_y'],
                'bluezone_radius': first_zone['bluezone_radius'],
                'whitezone_center_x': first_zone['whitezone_x'],
                'whitezone_center_y': first_zone['whitezone_y'],
                'whitezone_radius': first_zone['whitezone_radius']
            }
    
    def _calculate_squad_zone_distances_v2(self, squad_features, zone_data):
        """
        Squad 중심 위치 기반으로 BlueZone/WhiteZone 거리 계산 (V2)
        이미 계산된 positions를 활용하여 squad_features에 v2 컬럼 추가
        
        Args:
            squad_features: DataFrame with 'positions' column
            zone_data: dict with zone center and radius info
        
        Returns:
            squad_features with added 'dist_from_bluezone_v2' and 'dist_from_whitezone_v2' columns
        """
        def compute_squad_center(positions):
            """
            스쿼드 멤버들의 중심 좌표 계산
            positions: list of [x, y, z], 죽은 플레이어는 [0, 0, 0]
            """
            # [0, 0, 0]이 아닌 살아있는 플레이어만 필터링
            alive_positions = [pos for pos in positions if pos[0] != 0.0 or pos[1] != 0.0]
            
            if not alive_positions:
                return None
            
            # x, y 좌표의 평균
            center_x = np.mean([pos[0] for pos in alive_positions])
            center_y = np.mean([pos[1] for pos in alive_positions])
            
            return (center_x, center_y)
        
        def calculate_distance_ratio(x, y, center_x, center_y, radius):
            """중심으로부터의 거리 / 반경 비율 계산"""
            if radius <= 0:
                return -1.0
            
            distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
            return distance / radius
        
        # 각 squad의 중심 기반 거리 계산
        dist_bluezone_v2_list = []
        dist_whitezone_v2_list = []
        
        for _, row in squad_features.iterrows():
            positions = row['positions']
            
            # Squad 중심 계산
            squad_center = compute_squad_center(positions)
            
            if squad_center is None:
                # 모두 죽은 경우
                dist_bluezone_v2_list.append(-1.0)
                dist_whitezone_v2_list.append(-1.0)
            else:
                center_x, center_y = squad_center
                
                # BlueZone 거리 계산
                dist_blue = calculate_distance_ratio(
                    center_x, center_y,
                    zone_data['bluezone_center_x'],
                    zone_data['bluezone_center_y'],
                    zone_data['bluezone_radius']
                )
                
                # WhiteZone 거리 계산
                dist_white = calculate_distance_ratio(
                    center_x, center_y,
                    zone_data['whitezone_center_x'],
                    zone_data['whitezone_center_y'],
                    zone_data['whitezone_radius']
                )
                
                dist_bluezone_v2_list.append(dist_blue)
                dist_whitezone_v2_list.append(dist_white)
        
        # squad_features에 컬럼 추가
        squad_features['dist_from_bluezone_v2'] = dist_bluezone_v2_list
        squad_features['dist_from_whitezone_v2'] = dist_whitezone_v2_list
        
        return squad_features
    
    def _aggregate_squad_features(self, user_features):
        """
        Squad 내 user들의 feature를 집계
        """
        squad_features_list = []
        
        for squad_num in sorted(user_features['squad_number'].unique()):
            squad_users = user_features[user_features['squad_number'] == squad_num]
            
            # Action cumulative features: sum
            action_features = {}
            for feature in self.action_cumulative_features:
                if feature in squad_users.columns:
                    action_features[feature] = squad_users[feature].sum()
            
            # Snapshot features: mean (현재 시점의 평균값)
            snapshot_features = {}
            for feature in self.snapshot_features:
                if feature in squad_users.columns:
                    # 🎯 새로운 Team Status features는 특별 처리
                    if feature == 'current_health':
                        # 생존자들의 체력 총합만 계산 (is_alive = 1인 경우만)
                        if 'is_alive' in squad_users.columns:
                            alive_users = squad_users[squad_users['is_alive'] == 1]
                            snapshot_features['squad_total_health'] = alive_users[feature].sum()
                        else:
                            snapshot_features['squad_total_health'] = squad_users[feature].sum()
                    elif feature == 'is_alive':
                        # 생존자 수
                        snapshot_features['squad_alive_count'] = squad_users[feature].sum()
                    elif feature == 'is_groggy':
                        # groggy가 아닌 살아있는 유저 수
                        if 'is_alive' in squad_users.columns:
                            alive_users = squad_users[squad_users['is_alive'] == 1]
                            non_groggy_alive_users = alive_users[alive_users[feature] == 0]
                            snapshot_features['squad_non_groggy_count'] = len(non_groggy_alive_users)
                        else:
                            # is_alive가 없으면 groggy가 아닌 유저만 카운트
                            snapshot_features['squad_non_groggy_count'] = (squad_users[feature] == 0).sum()
                    elif feature == 'is_out_bluezone':
                        # 생존자 중 블루존 밖에 있는 유저 수
                        if 'is_alive' in squad_users.columns:
                            alive_users = squad_users[squad_users['is_alive'] == 1]
                            snapshot_features['squad_out_bluezone_count'] = alive_users[feature].sum()
                        else:
                            snapshot_features['squad_out_bluezone_count'] = squad_users[feature].sum()
                    elif feature == 'is_vehicle_ride':
                        # T시점 Vehicle 탑승 여부 팀원 수 (생존자만 sum)
                        if 'is_alive' in squad_users.columns:
                            alive_users = squad_users[squad_users['is_alive'] == 1]
                            snapshot_features['is_vehicle_count'] = alive_users[feature].sum() if not alive_users.empty else 0
                        else:
                            snapshot_features['is_vehicle_count'] = squad_users[feature].sum()
                    elif feature in ['dist_from_bluezone', 'dist_from_whitezone', 'n_teammates', 'n_enemies']:
                        # 생존자만 대상으로 평균 계산 (죽은 플레이어는 -1 값이므로 제외)
                        if 'is_alive' in squad_users.columns:
                            alive_users = squad_users[squad_users['is_alive'] == 1]
                            if not alive_users.empty:
                                # 생존자가 있으면 평균 계산 (-1 값 제외)
                                valid_values = alive_users[feature][alive_users[feature] >= 0]
                                snapshot_features[feature] = valid_values.mean() if not valid_values.empty else -1.0
                            else:
                                # 모두 죽었으면 -1
                                snapshot_features[feature] = -1.0
                        else:
                            # is_alive 정보가 없으면 -1 값만 제외하고 평균
                            valid_values = squad_users[feature][squad_users[feature] >= 0]
                            snapshot_features[feature] = valid_values.mean() if not valid_values.empty else -1.0
                    else:
                        # 🎯 아이템 개수는 생존자의 합계, 나머지는 평균
                        if 'is_alive' in squad_users.columns:
                            alive_users = squad_users[squad_users['is_alive'] == 1]
                            if not alive_users.empty:
                                # item_*_count 형태의 아이템 개수는 합계 (sum)
                                if feature.startswith('item_') and feature.endswith('_count'):
                                    snapshot_features[feature] = alive_users[feature].sum()
                                else:
                                    # 무기 점수, 방어구 레벨 등은 평균 (mean)
                                    snapshot_features[feature] = alive_users[feature].mean()
                            else:
                                # 모두 죽었으면 0으로 처리
                                snapshot_features[feature] = 0.0
                        else:
                            # is_alive 정보가 없으면 전체 평균/합계 (fallback)
                            if feature.startswith('item_') and feature.endswith('_count'):
                                snapshot_features[feature] = squad_users[feature].sum()
                            else:
                                snapshot_features[feature] = squad_users[feature].mean()
            
            # Squad 정보
            squad_info = {
                'squad_number': int(squad_num)
            }
            
            # 모든 feature 결합
            squad_feature = {**squad_info, **action_features, **snapshot_features}
            squad_features_list.append(squad_feature)
        
        return pd.DataFrame(squad_features_list)
    
    def generate_phase_samples(self, squad_labels=None, match_id=None):
        """
        각 phase별로 random time_point를 샘플링하고 squad feature 생성
        """
        all_samples = []
        all_time_points = []  # 모든 phase의 time_points를 먼저 수집 (다음 time_point 참조를 위해)
        phase_time_point_map = {}  # phase별 time_points 매핑
        
        for phase_num, config in self.phase_config.items():
            phase_start = config['start']
            phase_end = config['end']
            n_samples = config['n_samples']
            
            # Phase 내에서 random time_point 샘플링
            time_points = np.random.uniform(phase_start, phase_end, n_samples)
            time_points = np.sort(time_points)  # 시간 순으로 정렬
            
            phase_time_point_map[phase_num] = time_points
            all_time_points.extend(time_points)
        
        # 전체 time_points 정렬 (다음 time_point 찾기 위해)
        all_time_points = sorted(all_time_points)

        for phase_num, config in self.phase_config.items():
            time_points = phase_time_point_map[phase_num]
            
            for i, time_point in enumerate(time_points):
                # 해당 time_point까지의 squad feature 생성
                squad_features = self.get_squad_features_at_timepoint(time_point)
                
                # Phase 정보 추가
                squad_features['phase'] = phase_num
                squad_features['time_point'] = time_point
                squad_features['phase_sample_id'] = i + 1
                
                # Match ID 추가
                if match_id is not None:
                    squad_features['match_id'] = match_id
                
                # Squad 라벨 추가
                if squad_labels is not None:
                    squad_features = squad_features.merge(
                        squad_labels[['squad_number', 'squad_ranking', 'squad_win', 'squad_death_time']], 
                        on='squad_number', 
                        how='left'
                    )
                    
                    # 전체 squad 수 및 ranking ratio 추가
                    total_squad_cnt = len(squad_labels)
                    squad_features['total_squad_cnt'] = total_squad_cnt
                    squad_features['squad_ranking_relative'] = squad_features['squad_ranking'] / total_squad_cnt
                
                # 🎯 다음 time_point 찾기
                current_idx = all_time_points.index(time_point)
                next_time_point = all_time_points[current_idx + 1] if current_idx + 1 < len(all_time_points) else None
                
                # 🎯 Y Labels: squad_survival_time_diff, squad_survival_time_tf 계산
                if next_time_point is not None and squad_labels is not None:
                    # 시간 윈도우 길이 계산
                    time_window_length = next_time_point - time_point
                    squad_features['time_window_length'] = time_window_length
                    
                    def calculate_survival_labels(row):
                        squad_death_time = row['squad_death_time']
                        current_time = time_point
                        
                        # Squad가 다음 time_point까지 살아있는 경우
                        if pd.isna(squad_death_time) or squad_death_time >= next_time_point:
                            survival_time_diff = next_time_point - current_time
                            survival_tf = 1
                        # Squad가 그 사이에 죽은 경우
                        else:
                            survival_time_diff = max(squad_death_time - current_time, 0) # squad 죽으면 0으로 처리
                            survival_tf = 0
                        
                        # 상대적 생존 시간 비율 계산
                        survival_time_diff_relative = survival_time_diff / time_window_length if time_window_length > 0 else 0
                        
                        return pd.Series({
                            'squad_survival_time_diff': survival_time_diff,
                            'squad_survival_time_tf': survival_tf,
                            'squad_survival_time_diff_relative': survival_time_diff_relative
                        })
                    
                    survival_labels = squad_features.apply(calculate_survival_labels, axis=1)
                    squad_features['squad_survival_time_diff'] = survival_labels['squad_survival_time_diff']
                    squad_features['squad_survival_time_tf'] = survival_labels['squad_survival_time_tf'].astype(int)
                    squad_features['squad_survival_time_diff_relative'] = survival_labels['squad_survival_time_diff_relative']
                else:
                    # 마지막 time_point이거나 squad_labels가 없는 경우
                    squad_features['time_window_length'] = np.nan
                    squad_features['squad_survival_time_diff'] = np.nan
                    squad_features['squad_survival_time_tf'] = np.nan
                    squad_features['squad_survival_time_diff_relative'] = np.nan
                
                all_samples.append(squad_features)
        
        # 모든 샘플 결합
        final_dataset = pd.concat(all_samples, ignore_index=True)
        
        # 🎯 Dtype 변환 적용
        final_dataset = self._convert_dtypes(final_dataset)

        basic_cols = ['match_id', 'squad_number', 'phase', 'phase_sample_id', 'time_point', 'total_squad_cnt', 'time_window_length']
        label_cols = ['squad_ranking', 'squad_win', 'squad_death_time', 'squad_ranking_relative', 'squad_survival_time_diff', 'squad_survival_time_tf', 'squad_survival_time_diff_relative']
        # ingame features (기본/라벨 컬럼 제외한 나머지)
        feature_cols = [col for col in final_dataset.columns if col not in basic_cols + label_cols]

        # 최종 컬럼 순서
        final_columns = basic_cols + feature_cols + label_cols
        
        # 존재하는 컬럼만 선택
        final_columns = [col for col in final_columns if col in final_dataset.columns]
        
        return final_dataset[final_columns]

    def generate_phase_samples_w_time_interval(self, squad_labels=None, match_id=None, time_interval=100, start_time=100, end_time=2000):
        """
        고정된 시간 간격(time_interval)으로 time_point를 샘플링하고 squad feature 생성
        
        Args:
            squad_labels: Squad 라벨 정보
            match_id: 매치 ID
            time_interval: 샘플링 시간 간격 (초 단위, 기본값: 100)
            start_time: 샘플링 시작 시간 (기본값: 100초)
            end_time: 샘플링 종료 시간 (기본값: 2000초)
        """
        all_samples = []
        
        # 고정된 간격으로 time_points 생성 (예: 100, 200, 300, ..., 2000)
        all_time_points = list(range(start_time, end_time + 1, time_interval))
        
        for time_point in all_time_points:
            # 해당 time_point까지의 squad feature 생성
            squad_features = self.get_squad_features_at_timepoint(time_point)
            
            # Phase 정보 추가 (어느 phase에 속하는지 계산)
            phase_num = self._get_phase_for_timepoint(time_point)
            squad_features['phase'] = phase_num
            squad_features['time_point'] = time_point
            
            # Match ID 추가
            if match_id is not None:
                squad_features['match_id'] = match_id
            
            # Squad 라벨 추가
            if squad_labels is not None:
                squad_features = squad_features.merge(
                    squad_labels[['squad_number', 'squad_ranking', 'squad_win', 'squad_death_time']], 
                    on='squad_number', 
                    how='left'
                )
                
                # 전체 squad 수 및 ranking ratio 추가
                total_squad_cnt = len(squad_labels)
                squad_features['total_squad_cnt'] = total_squad_cnt
                squad_features['squad_ranking_relative'] = squad_features['squad_ranking'] / total_squad_cnt
            
            # 🎯 다음 time_point 찾기
            current_idx = all_time_points.index(time_point)
            next_time_point = all_time_points[current_idx + 1] if current_idx + 1 < len(all_time_points) else None
            
            # 🎯 Y Labels: squad_survival_time_diff, squad_survival_time_tf 계산
            if next_time_point is not None and squad_labels is not None:
                # 시간 윈도우 길이 계산
                time_window_length = next_time_point - time_point
                squad_features['time_window_length'] = time_window_length
                
                def calculate_survival_labels(row):
                    squad_death_time = row['squad_death_time']
                    current_time = time_point
                    
                    # Squad가 다음 time_point까지 살아있는 경우
                    if pd.isna(squad_death_time) or squad_death_time >= next_time_point:
                        survival_time_diff = next_time_point - current_time
                        survival_tf = 1
                    # Squad가 그 사이에 죽은 경우
                    else:
                        survival_time_diff = max(squad_death_time - current_time, 0) # squad 죽으면 0으로 처리
                        survival_tf = 0
                    
                    # 상대적 생존 시간 비율 계산
                    survival_time_diff_relative = survival_time_diff / time_window_length if time_window_length > 0 else 0
                    
                    return pd.Series({
                        'squad_survival_time_diff': survival_time_diff,
                        'squad_survival_time_tf': survival_tf,
                        'squad_survival_time_diff_relative': survival_time_diff_relative
                    })
                
                survival_labels = squad_features.apply(calculate_survival_labels, axis=1)
                squad_features['squad_survival_time_diff'] = survival_labels['squad_survival_time_diff']
                squad_features['squad_survival_time_tf'] = survival_labels['squad_survival_time_tf'].astype(int)
                squad_features['squad_survival_time_diff_relative'] = survival_labels['squad_survival_time_diff_relative']
            else:
                # 마지막 time_point이거나 squad_labels가 없는 경우
                squad_features['time_window_length'] = np.nan
                squad_features['squad_survival_time_diff'] = np.nan
                squad_features['squad_survival_time_tf'] = np.nan
                squad_features['squad_survival_time_diff_relative'] = np.nan
            
            all_samples.append(squad_features)
        
        # 모든 샘플 결합
        final_dataset = pd.concat(all_samples, ignore_index=True)
        
        # 🎯 Dtype 변환 적용
        final_dataset = self._convert_dtypes(final_dataset)

        basic_cols = ['match_id', 'squad_number', 'phase', 'time_point', 'total_squad_cnt', 'time_window_length']
        label_cols = ['squad_ranking', 'squad_win', 'squad_death_time', 'squad_ranking_relative', 'squad_survival_time_diff', 'squad_survival_time_tf', 'squad_survival_time_diff_relative']
        # ingame features (기본/라벨 컬럼 제외한 나머지)
        feature_cols = [col for col in final_dataset.columns if col not in basic_cols + label_cols]

        # 최종 컬럼 순서
        final_columns = basic_cols + feature_cols + label_cols
        
        # 존재하는 컬럼만 선택
        final_columns = [col for col in final_columns if col in final_dataset.columns]
        
        return final_dataset[final_columns]
    
    def _get_phase_for_timepoint(self, time_point):
        """
        특정 time_point가 어느 phase에 속하는지 계산
        """
        for phase_num, config in self.phase_config.items():
            if config['start'] <= time_point < config['end']:
                return phase_num
        # phase 범위를 벗어나면 가장 가까운 phase 반환
        if time_point < self.phase_config[1]['start']:
            return 1
        return max(self.phase_config.keys())
    
    def _convert_dtypes(self, df):
        """
        최종 데이터셋의 dtype을 올바르게 변환
        """
        df = df.copy()
        
        # 1. n_ prefix 컬럼 → int (n_teammates, n_enemies 제외 - float 유지)
        n_prefix_cols = [col for col in df.columns if col.startswith('n_')]
        exclude_cols = ['n_teammates', 'n_enemies']
        n_prefix_cols = [col for col in n_prefix_cols if col not in exclude_cols]
        
        for col in n_prefix_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)
        
        # 2. item_*_count 컬럼 → int
        item_count_cols = [col for col in df.columns 
                           if col.startswith('item_') and col.endswith('_count')]
        for col in item_count_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)
        
        # 3. match_date → string (존재하는 경우)
        if 'match_date' in df.columns:
            df['match_date'] = df['match_date'].astype(str)
        
        # 4. positions: list
        # 5. bluezone_info, whitezone_info: float tuple
        # 6. item_*_level: float
        
        return df
        
    def feature_death_time(self, parsed_df, end_time, start_time=1):
        return parsed_df['death_time'].apply(
            lambda x: end_time if x is None else (x if start_time <= x < end_time else end_time)
        )
    
    def feature_n_pickup(self, parsed_df, end_time, start_time=1):
        return parsed_df['pickup_time'].apply(lambda list_x: \
            len([t for t in list_x if (start_time <= t) & (t < end_time)])
            if notNone(list_x) else 0)
    
    def feature_n_drop(self, parsed_df, end_time, start_time=1):
        return parsed_df['drop_time'].apply(lambda list_x: \
            len([t for t in list_x if (start_time <= t) & (t < end_time)])
            if notNone(list_x) else 0)
    
    def feature_n_teammates_at_timepoint(self, parsed_df, time_point):
        """특정 time_point에서의 근처 팀원 수 (snapshot)"""
        def get_teammates_at_time(row):
             # 플레이어가 죽었으면 -1 반환
            death_time = row.get('death_time')
            if death_time is not None and death_time <= time_point:
                return -1
            
            count_times = row.get('count_time', [])
            num_teammates = row.get('num_near_teammates', [])
            
            if not notNone(count_times) or not notNone(num_teammates):
                return 0
            
            # time_point 이전의 가장 최근 값 찾기
            valid_indices = [i for i, t in enumerate(count_times) if t <= time_point]
            if valid_indices:
                # 🎯 시간이 가장 큰 인덱스 찾기
                latest_idx = max(valid_indices, key=lambda i: count_times[i])
                return int(num_teammates[latest_idx])
            else:
                return 0
        
        return parsed_df.apply(get_teammates_at_time, axis=1)
    
    def feature_n_enemies_at_timepoint(self, parsed_df, time_point):
        """특정 time_point에서의 근처 적 수 (snapshot)"""
        def get_enemies_at_time(row):
            # 플레이어가 죽었으면 -1 반환
            death_time = row.get('death_time')
            if death_time is not None and death_time <= time_point:
                return -1
            
            count_times = row.get('count_time', [])
            num_enemies = row.get('num_near_enemies', [])
            
            if not notNone(count_times) or not notNone(num_enemies):
                return 0
            
            # time_point 이전의 가장 최근 값 찾기
            valid_indices = [i for i, t in enumerate(count_times) if t <= time_point]
            if valid_indices:
                # 🎯 시간이 가장 큰 인덱스 찾기
                latest_idx = max(valid_indices, key=lambda i: count_times[i])
                return int(num_enemies[latest_idx])
            else:
                return 0
        
        return parsed_df.apply(get_enemies_at_time, axis=1)
    
    def feature_dist_from_bluezone_at_timepoint(self, parsed_df, time_point):
        """특정 time_point에서의 BlueZone(현재 자기장) 중심으로부터의 거리 비율 (snapshot)"""
        def get_dist_at_time(row):
            # 🔥 플레이어가 죽었으면 -1 반환
            death_time = row.get('death_time')
            if death_time is not None and death_time <= time_point:
                return -1.0
            
            state_times = row.get('state_time', [])
            dist_from_bluezone = row.get('dist_from_bluezone', [])
            
            if not notNone(state_times) or not notNone(dist_from_bluezone):
                return 0.0
            
            # time_point 이전의 가장 최근 값 찾기
            valid_indices = [i for i, t in enumerate(state_times) if t <= time_point]
            if valid_indices:
                # 🎯 시간이 가장 큰 인덱스 찾기
                latest_idx = max(valid_indices, key=lambda i: state_times[i])
                value = float(dist_from_bluezone[latest_idx])
                # 🎯 원본 데이터가 -1이면 zone 정보가 없다는 의미이므로 0.0 반환
                return 0.0 if value < 0 else value
            else:
                return 0.0
        
        return parsed_df.apply(get_dist_at_time, axis=1)
    
    def feature_dist_from_whitezone_at_timepoint(self, parsed_df, time_point):
        """특정 time_point에서의 WhiteZone(다음 안전지대) 중심으로부터의 거리 비율 (snapshot)"""
        def get_dist_at_time(row):
            # 🔥 플레이어가 죽었으면 -1 반환
            death_time = row.get('death_time')
            if death_time is not None and death_time <= time_point:
                return -1.0
            
            state_times = row.get('state_time', [])
            dist_from_whitezone = row.get('dist_from_whitezone', [])
            
            if not notNone(state_times) or not notNone(dist_from_whitezone):
                return 0.0
            
            # time_point 이전의 가장 최근 값 찾기
            valid_indices = [i for i, t in enumerate(state_times) if t <= time_point]
            if valid_indices:
                # 🎯 시간이 가장 큰 인덱스 찾기
                latest_idx = max(valid_indices, key=lambda i: state_times[i])
                value = float(dist_from_whitezone[latest_idx])
                # 🎯 원본 데이터가 -1이면 zone 정보가 없다는 의미이므로 0.0 반환
                return 0.0 if value < 0 else value
            else:
                return 0.0
        
        return parsed_df.apply(get_dist_at_time, axis=1)
    
    
    def feature_all_dmg(self, parsed_df, isVictim, end_time, start_time=1):
        """
        모든 무기의 데미지를 통합하여 계산 (무기 종류 구분 없음)
        """
        def count_dmg(td_attack_time, td_damage, end_time, start_time):
            if notNone(td_attack_time):
                n_dmg = 0
                damages = []
                for t, dmg in zip(td_attack_time, td_damage):
                    if (start_time < t) & (t < end_time):    
                        if dmg > 0:
                            n_dmg = n_dmg + 1
                            damages.append(dmg)
                return pd.Series([n_dmg, np.sum(damages)], index=['n_dmg', 'sum_dmg'])
            else:
                return pd.Series([0, 0], index=['n_dmg', 'sum_dmg'])
            
        if isVictim:
            return parsed_df[['td_victim_ingame_ed', 'td_victim_damage']].apply(lambda row: 
                count_dmg(row.td_victim_ingame_ed, row.td_victim_damage, end_time, start_time), axis=1)
        else:               
            return parsed_df[['td_attack_ingame_ed', 'td_attack_damage']].apply(lambda row: 
                count_dmg(row.td_attack_ingame_ed, row.td_attack_damage, end_time, start_time), axis=1)
    
    def feature_dmg_by_weapon(self, parsed_df, weapon_type, end_time, start_time=1):
        """
        특정 무기 타입별 데미지 계산 (AR, SR, DMR, SMG, SHOTGUN)
        """
        # 무기 타입에 따른 weapon_group 설정 (td_* 버전 사용)
        if weapon_type == 'AR':
            weapon_list = td_AR_WEAPONS
        elif weapon_type == 'DMR':
            weapon_list = td_DMR_WEAPONS
        elif weapon_type == 'SR':
            weapon_list = td_SR_WEAPONS
        elif weapon_type == 'SMG':
            weapon_list = td_SMG_WEAPONS
        elif weapon_type == 'SHOTGUN':
            weapon_list = td_SHOTGUN_WEAPONS
        else:
            raise ValueError(f"Unsupported weapon_type: {weapon_type}")
        
        # 환경 피해 및 투척물 등 제외할 무기 패턴
        excluded_patterns = ['BP_', 'Proj', 'Debuff', 'Fire']
        
        def filter_dmg(td_attack_time, td_weapon, td_damage, weapon_list, end_time, start_time):
            if notNone(td_attack_time) and notNone(td_weapon):
                n_dmg = 0
                damages = []
                for t, weapon, dmg in zip(td_attack_time, td_weapon, td_damage):
                    # 환경 피해/투척물 제외
                    if any(pattern in weapon for pattern in excluded_patterns):
                        continue
                    
                    # weapon 이름 정규화: WeapAK47_C → AK47
                    weapon_short = weapon.split('_')[0].replace('Weap', '')
                    if (start_time < t) & (t < end_time):
                        if weapon_short in weapon_list:
                            if dmg > 0:
                                n_dmg = n_dmg + 1
                                damages.append(dmg)
                return pd.Series([n_dmg, np.sum(damages)], index=[f'n_dmg_by_{weapon_type}', f'sum_dmg_by_{weapon_type}'])
            else:
                return pd.Series([0, 0], index=[f'n_dmg_by_{weapon_type}', f'sum_dmg_by_{weapon_type}'])
        
        return parsed_df[['td_attack_ingame_ed', 'td_attack_weapon', 'td_attack_damage']].apply(
            lambda row: filter_dmg(row.td_attack_ingame_ed, row.td_attack_weapon, row.td_attack_damage, 
                                   weapon_list, end_time, start_time), axis=1)
    
    def feature_kill_stats(self, parsed_df, stat_name, end_time, start_time):  # kill, finish, groggy, squad_last_member_kill
        # 🎯 groggy는 LogPlayerMakeGroggy에서 추출한 dbno_maker_time 사용 (모든 기절 포함)
        # kill, finish, squad_last_member_kill은 LogPlayerKillV2에서 추출한 killv2_*_time 사용
        if stat_name == 'groggy':
            col_name = 'dbno_maker_time'
        else:
            col_name = f'killv2_{stat_name}_time'
        
        return parsed_df[[col_name]].apply(lambda row: \
            len([t for t in row[col_name] if (start_time <= t) & (t < end_time)]) 
            if notNone(row[col_name]) else 0, axis=1)
    
    def feature_current_health(self, parsed_df, time_point):
        """
        특정 time_point에서의 현재 체력 (bluechip revival 반영)
        🎯 survival_time (KILLED) + revival_time (REVIVED) 통합 사용
        """
        def get_health_at_timepoint(row):
            health_times = row.get('health_time', [])
            health_values = row.get('health_value', [])
            
            # 🎯 survival_time에서 KILLED 이벤트 가져오기
            survival_times = row.get('survival_time', [])
            if survival_times is None or (isinstance(survival_times, float) and pd.isna(survival_times)):
                survival_times = []
            
            # 🎯 revival_time에서 REVIVED 이벤트 가져오기
            revival_times = row.get('revival_time', [])
            if revival_times is not None and not isinstance(revival_times, (list, np.ndarray, pd.Series)):
                revival_times = [revival_times] if not pd.isna(revival_times) else []
            if revival_times is None or (isinstance(revival_times, float) and pd.isna(revival_times)):
                revival_times = []
            
            # 🎯 두 이벤트 리스트 통합 후 시간순 정렬
            life_death_events = []
            for t in survival_times:
                life_death_events.append((t, 'KILLED'))
            for t in revival_times:
                life_death_events.append((t, 'REVIVED'))
            life_death_events.sort(key=lambda x: x[0])
            
            # 🎯 1단계: time_point 시점의 생존 여부 확인
            is_alive = True
            last_revival_time = None
            for event_time, event_type in life_death_events:
                if event_time <= time_point:
                    if event_type == 'KILLED':
                        is_alive = False
                        last_revival_time = None  # 죽으면 revival 상태 리셋
                    elif event_type == 'REVIVED':
                        is_alive = True
                        last_revival_time = event_time
            
            # 죽어있으면 체력 0
            if not is_alive:
                return 0.0
            
            # 🎯 2단계: 체력 계산
            # Bluechip revival 직후면 체력 100
            if last_revival_time is not None:
                # Revival 이후의 체력 변화 확인
                if health_times and health_values:
                    post_revival_healths = [(t, h) for t, h in zip(health_times, health_values) 
                                           if last_revival_time <= t <= time_point]
                    if post_revival_healths:
                        # 🎯 Revival 이후 체력 변화가 있으면 최신 시간의 값 사용
                        latest_health = max(post_revival_healths, key=lambda x: x[0])
                        return latest_health[1]
                    else:
                        # Revival 직후 체력 변화가 없으면 100
                        return 100.0
            
            # 일반적인 체력 조회
            if health_times and health_values:
                valid_indices = [i for i, t in enumerate(health_times) if t <= time_point]
                if valid_indices:
                    # 🎯 시간이 가장 큰 인덱스 찾기 (정렬 안 된 경우 대비)
                    latest_idx = max(valid_indices, key=lambda i: health_times[i])
                    return health_values[latest_idx]
            
            return 100.0  # 기본 체력
        
        return parsed_df.apply(get_health_at_timepoint, axis=1)
    
    def feature_is_alive(self, parsed_df, time_point):
        """
        특정 time_point에서의 생존 여부 (1: 생존, 0: 사망)
        🎯 survival_time (KILLED) + revival_time (REVIVED) 통합 사용
        """
        def check_alive_at_timepoint(row):
            # 🎯 survival_time에서 KILLED 이벤트 가져오기
            survival_times = row.get('survival_time', [])
            if survival_times is None or (isinstance(survival_times, float) and pd.isna(survival_times)):
                survival_times = []
            
            # 🎯 revival_time에서 REVIVED 이벤트 가져오기
            revival_times = row.get('revival_time', [])
            if revival_times is not None and not isinstance(revival_times, (list, np.ndarray, pd.Series)):
                revival_times = [revival_times] if not pd.isna(revival_times) else []
            if revival_times is None or (isinstance(revival_times, float) and pd.isna(revival_times)):
                revival_times = []
            
            # 생사 이벤트가 없으면 살아있음
            if not survival_times and not revival_times:
                return 1
            
            # 🎯 두 이벤트 리스트 통합 후 시간순 정렬
            life_death_events = []
            for t in survival_times:
                life_death_events.append((t, 'KILLED'))
            for t in revival_times:
                life_death_events.append((t, 'REVIVED'))
            life_death_events.sort(key=lambda x: x[0])
            
            # time_point까지의 이벤트를 순차적으로 처리
            is_alive = True
            for event_time, event_type in life_death_events:
                if event_time <= time_point:
                    if event_type == 'KILLED':
                        is_alive = False
                    elif event_type == 'REVIVED':
                        is_alive = True
            
            return 1 if is_alive else 0
        
        return parsed_df.apply(check_alive_at_timepoint, axis=1)
    
    def feature_is_out_bluezone(self, parsed_df, time_point):
        """특정 time_point에서 블루존 밖에 있는지 여부 (1: 블루존 밖, 0: 블루존 안)"""
        def check_out_bluezone_at_timepoint(row):
            health_times = row.get('health_time', [])
            health_in_bluezone = row.get('health_in_bluezone', [])
            
            if not health_times or not health_in_bluezone:
                return 0  # 기본적으로 블루존 안
            
            # time_point 이전의 최신 블루존 상태 정보
            valid_indices = [i for i, t in enumerate(health_times) if t <= time_point]
            if valid_indices:
                # 🎯 시간이 가장 큰 인덱스 찾기
                latest_idx = max(valid_indices, key=lambda i: health_times[i])
                # health_in_bluezone이 True면 블루존 밖 (1), False면 블루존 안 (0)
                return 1 if health_in_bluezone[latest_idx] else 0
            else:
                return 0  # 기본적으로 블루존 안
        
        return parsed_df.apply(check_out_bluezone_at_timepoint, axis=1)
    
    def feature_is_groggy(self, parsed_df, time_point):
        """특정 time_point에서 groggy 상태인지 여부 (1: groggy, 0: 정상)"""
        def check_groggy_at_timepoint(row):
            groggy_times = row.get('groggy_status_time', [])
            groggy_events = row.get('groggy_status_event', [])
            
            # 스칼라 값인 경우 리스트로 변환
            if groggy_times is not None and not isinstance(groggy_times, (list, np.ndarray, pd.Series)):
                groggy_times = [groggy_times] if not pd.isna(groggy_times) else []
            if groggy_events is not None and not isinstance(groggy_events, (list, np.ndarray, pd.Series)):
                groggy_events = [groggy_events] if not pd.isna(groggy_events) else []
            
            # None/NaN 처리
            if groggy_times is None or (isinstance(groggy_times, float) and pd.isna(groggy_times)):
                groggy_times = []
            if groggy_events is None or (isinstance(groggy_events, float) and pd.isna(groggy_events)):
                groggy_events = []
            
            if not groggy_times or not groggy_events:
                return 0  # groggy 이벤트가 없으면 정상
            
            # 시간 순서대로 groggy 상태 추적
            is_groggy = False
            try:
                for event_time, event_type in zip(groggy_times, groggy_events):
                    if event_time <= time_point:
                        if event_type == 'GROGGY_START':
                            is_groggy = True
                        elif event_type == 'GROGGY_END':
                            is_groggy = False
                return 1 if is_groggy else 0
            except TypeError:
                return 0
        
        return parsed_df.apply(check_groggy_at_timepoint, axis=1)
    
    def feature_n_heal(self, parsed_df, time_point):
        """특정 time_point까지의 총 힐 아이템 사용 횟수 (LogItemUse 기반 - bandage + firstaid + medkit)"""
        # 🎯 LogHeal은 여러 번 찍히는 문제가 있어서 LogItemUse 사용
        heal_item_ids = [
            'Item_Heal_Bandage_C',
            'Item_Heal_FirstAid_C',
            'Item_Heal_MedKit_C'
        ]
        
        def count_heal(row):
            use_times = row.get('use_time', [])
            use_items = row.get('use_item', [])
            if not notNone(use_times) or not notNone(use_items):
                return 0
            count = sum([1 for t, item in zip(use_times, use_items) 
                        if t <= time_point and item in heal_item_ids])
            return count
        return parsed_df.apply(count_heal, axis=1)
    
    def feature_sum_heal_amount(self, parsed_df, time_point):
        """특정 time_point까지의 총 회복량 (부스트 효과 포함)"""
        def sum_heal_amount(row):
            heal_times = row.get('heal_time', [])
            heal_amounts = row.get('heal_amount', [])
            if not notNone(heal_times) or not notNone(heal_amounts):
                return 0
            # 🎯 모든 회복량 합산 (부스트 아이템의 시간당 회복 포함)
            # 부스트 효과는 itemId가 빈 문자열로 기록됨
            total = 0
            for t, amt in zip(heal_times, heal_amounts):
                if t <= time_point:
                    total += amt
            return total
        return parsed_df.apply(sum_heal_amount, axis=1)
    
    def feature_n_heal_by_item(self, parsed_df, time_point, item_name='bandage'):
        """특정 time_point까지 특정 힐 아이템 사용 횟수 (LogItemUse 기반)"""
        # 🎯 LogHeal은 여러 번 찍히는 문제가 있어서 LogItemUse 사용
        item_mapping = {
            'bandage': 'Item_Heal_Bandage_C',
            'firstaid': 'Item_Heal_FirstAid_C',
            'medkit': 'Item_Heal_MedKit_C',
        }
        target_item = item_mapping.get(item_name, item_name)
        
        def count_item_heal(row):
            use_times = row.get('use_time', [])
            use_items = row.get('use_item', [])
            if not notNone(use_times) or not notNone(use_items):
                return 0
            count = sum([1 for t, item in zip(use_times, use_items) 
                        if t <= time_point and item == target_item])
            return count
        return parsed_df.apply(count_item_heal, axis=1)
    
    def feature_n_boost_by_item(self, parsed_df, time_point, item_name='painkiller'):
        """특정 time_point까지 특정 부스트 아이템 사용 횟수 (LogItemUse 기반)"""
        item_mapping = {
            'painkiller': 'Item_Boost_PainKiller_C',
            'energydrink': 'Item_Boost_EnergyDrink_C',
            'adrenaline': 'Item_Boost_AdrenalineSyringe_C'
        }
        target_item = item_mapping.get(item_name, item_name)
        
        def count_item_use(row):
            use_times = row.get('use_time', [])
            use_items = row.get('use_item', [])
            if not notNone(use_times) or not notNone(use_items):
                return 0
            count = sum([1 for t, item in zip(use_times, use_items) 
                        if t <= time_point and item == target_item])
            return count
        return parsed_df.apply(count_item_use, axis=1)
    
    def feature_vehicle_has(self, parsed_df, time_point):
        """특정 time_point에서 vehicle 보유 여부 (0/1)"""
        def has_vehicle(row):
            vehicle_times = row.get('vehicle_time', [])
            vehicle_usetypes = row.get('vehicle_usetype', [])
            
            if not notNone(vehicle_times) or not notNone(vehicle_usetypes):
                return 0
            
            # time_point 이전의 이벤트만 필터링
            has_vehicle_status = False
            for t, use_type in zip(vehicle_times, vehicle_usetypes):
                if t <= time_point:
                    if use_type == 'ride':
                        has_vehicle_status = True
                    elif use_type == 'leave':
                        has_vehicle_status = False
            
            return 1 if has_vehicle_status else 0
        return parsed_df.apply(has_vehicle, axis=1)
    
    def feature_n_vehicle_ride(self, parsed_df, time_point):
        """특정 time_point까지의 총 vehicle 탑승 횟수"""
        def count_rides(row):
            vehicle_times = row.get('vehicle_time', [])
            vehicle_usetypes = row.get('vehicle_usetype', [])
            
            if not notNone(vehicle_times) or not notNone(vehicle_usetypes):
                return 0
            
            count = sum([1 for t, use_type in zip(vehicle_times, vehicle_usetypes) 
                        if t <= time_point and use_type == 'ride'])
            return count
        return parsed_df.apply(count_rides, axis=1)
    
    def feature_item_helmet_level(self, parsed_df, time_point):
        """특정 time_point에서 장착한 헬멧 레벨 (0, 1, 2, 3)"""
        def get_helmet_level(row):
            equip_times = row.get('equip_time', [])
            equip_items = row.get('equip_item', [])
            
            if not notNone(equip_times) or not notNone(equip_items):
                return 0
            
            # time_point 이전의 최신 헬멧 찾기
            current_level = 0
            for t, item in zip(equip_times, equip_items):
                if t <= time_point and item in HELMET_ITEMS:
                    current_level = HELMET_ITEMS[item]
            
            return current_level
        return parsed_df.apply(get_helmet_level, axis=1)
    
    def feature_item_vest_level(self, parsed_df, time_point):
        """특정 time_point에서 장착한 조끼 레벨 (0, 1, 2, 3)"""
        def get_vest_level(row):
            equip_times = row.get('equip_time', [])
            equip_items = row.get('equip_item', [])
            
            if not notNone(equip_times) or not notNone(equip_items):
                return 0
            
            # time_point 이전의 최신 조끼 찾기
            current_level = 0
            for t, item in zip(equip_times, equip_items):
                if t <= time_point and item in VEST_ITEMS:
                    current_level = VEST_ITEMS[item]
            
            return current_level
        return parsed_df.apply(get_vest_level, axis=1)
    
    def feature_item_backpack_level(self, parsed_df, time_point):
        """특정 time_point에서 장착한 백팩 레벨 (0, 1, 2, 3)"""
        def get_backpack_level(row):
            equip_times = row.get('equip_time', [])
            equip_items = row.get('equip_item', [])
            
            if not notNone(equip_times) or not notNone(equip_items):
                return 0
            
            # time_point 이전의 최신 백팩 찾기
            current_level = 0
            for t, item in zip(equip_times, equip_items):
                if t <= time_point and item in BACKPACK_ITEMS:
                    current_level = BACKPACK_ITEMS[item]
            
            return current_level
        return parsed_df.apply(get_backpack_level, axis=1)
    
    def feature_n_throwables(self, parsed_df, time_point, item_type='total'):
        """특정 time_point에서 보유한 투척 아이템 개수"""
        # item_type: 'total', 'grenade', 'smoke', 'flashbang', 'molotov'
        
        if item_type == 'total':
            target_items = list(THROWABLE_ITEMS.keys())
        else:
            target_items = [k for k, v in THROWABLE_ITEMS.items() if v == item_type]
        
        def count_throwables(row):
            pickup_times = row.get('pickup_time', [])
            pickup_items = row.get('pickup_item', [])
            pickup_stacks = row.get('pickup_stack', [])  # 🎯 stackCount
            drop_times = row.get('drop_time', [])
            drop_items = row.get('drop_item', [])
            drop_stacks = row.get('drop_stack', [])  # 🎯 stackCount
            # 🎯 LogPlayerUseThrowable에서 파싱된 투척 아이템 사용 정보 사용
            throwable_use_times = row.get('throwable_use_time', [])
            throwable_use_items = row.get('throwable_use_item', [])
            
            # 각 아이템별 개수 추적
            item_counts = {item: 0 for item in target_items}
            
            # Pickup 이벤트 처리 (stackCount 반영)
            if notNone(pickup_times) and notNone(pickup_items):
                # pickup_stacks가 없을 수도 있으므로 체크
                if notNone(pickup_stacks) and len(pickup_stacks) == len(pickup_items):
                    for t, item, stack in zip(pickup_times, pickup_items, pickup_stacks):
                        if t <= time_point and item in target_items:
                            # 🎯 stackCount가 0 또는 None이면 1로 처리
                            item_counts[item] += int(stack) if stack else 1
                else:
                    # stackCount 정보가 없으면 1개씩
                    for t, item in zip(pickup_times, pickup_items):
                        if t <= time_point and item in target_items:
                            item_counts[item] += 1
            
            # Drop 이벤트 처리 (stackCount 반영)
            if notNone(drop_times) and notNone(drop_items):
                if notNone(drop_stacks) and len(drop_stacks) == len(drop_items):
                    for t, item, stack in zip(drop_times, drop_items, drop_stacks):
                        if t <= time_point and item in target_items:
                            # 🎯 stackCount가 0 또는 None이면 1로 처리
                            item_counts[item] = max(0, item_counts[item] - (int(stack) if stack else 1))
                else:
                    # stackCount 정보가 없으면 1개씩
                    for t, item in zip(drop_times, drop_items):
                        if t <= time_point and item in target_items:
                            item_counts[item] = max(0, item_counts[item] - 1)
            
            # 🎯 Throwable Use 이벤트 처리 (LogPlayerUseThrowable 기반)
            # LogPlayerUseThrowable과 LogItemPickup 모두 같은 형식 사용 (Item_Weapon_XXX_C)
            # 사용은 항상 1개씩
            if notNone(throwable_use_times) and notNone(throwable_use_items):
                for t, item in zip(throwable_use_times, throwable_use_items):
                    if t <= time_point and item in target_items:
                        item_counts[item] = max(0, item_counts[item] - 1)
            
            return sum(item_counts.values())
        
        return parsed_df.apply(count_throwables, axis=1)
    
    def feature_n_aids(self, parsed_df, time_point, item_type='total'):
        """특정 time_point에서 보유한 회복 아이템 개수"""
        # item_type: 'total', 'bandage', 'firstaid', 'medkit', 'painkiller', 'energydrink', 'adrenaline'
        
        if item_type == 'total':
            target_items = list(AIDS_ITEMS.keys())
        else:
            target_items = [k for k, v in AIDS_ITEMS.items() if v == item_type]
        
        def count_aids(row):
            pickup_times = row.get('pickup_time', [])
            pickup_items = row.get('pickup_item', [])
            pickup_stacks = row.get('pickup_stack', [])  # 🎯 stackCount
            drop_times = row.get('drop_time', [])
            drop_items = row.get('drop_item', [])
            drop_stacks = row.get('drop_stack', [])  # 🎯 stackCount
            heal_times = row.get('heal_time', [])
            heal_items = row.get('heal_item', [])
            use_times = row.get('use_time', [])
            use_items = row.get('use_item', [])
            
            # 각 아이템별 개수 추적
            item_counts = {item: 0 for item in target_items}
            
            # Pickup 이벤트 처리 (stackCount 반영)
            if notNone(pickup_times) and notNone(pickup_items):
                if notNone(pickup_stacks) and len(pickup_stacks) == len(pickup_items):
                    for t, item, stack in zip(pickup_times, pickup_items, pickup_stacks):
                        if t <= time_point and item in target_items:
                            # 🎯 stackCount가 0 또는 None이면 1로 처리
                            item_counts[item] += int(stack) if stack else 1
                else:
                    for t, item in zip(pickup_times, pickup_items):
                        if t <= time_point and item in target_items:
                            item_counts[item] += 1
            
            # Drop 이벤트 처리 (stackCount 반영)
            if notNone(drop_times) and notNone(drop_items):
                if notNone(drop_stacks) and len(drop_stacks) == len(drop_items):
                    for t, item, stack in zip(drop_times, drop_items, drop_stacks):
                        if t <= time_point and item in target_items:
                            # 🎯 stackCount가 0 또는 None이면 1로 처리
                            item_counts[item] = max(0, item_counts[item] - (int(stack) if stack else 1))
                else:
                    for t, item in zip(drop_times, drop_items):
                        if t <= time_point and item in target_items:
                            item_counts[item] = max(0, item_counts[item] - 1)
            
            # Heal 이벤트 처리 (사용 시 -1, 항상 1개씩)
            if notNone(heal_times) and notNone(heal_items):
                for t, item in zip(heal_times, heal_items):
                    if t <= time_point and item in target_items:
                        item_counts[item] = max(0, item_counts[item] - 1)
            
            # Use 이벤트 처리 (부스트 아이템 사용 시 -1, 항상 1개씩)
            if notNone(use_times) and notNone(use_items):
                for t, item in zip(use_times, use_items):
                    if t <= time_point and item in target_items:
                        item_counts[item] = max(0, item_counts[item] - 1)
            
            return sum(item_counts.values())
        
        return parsed_df.apply(count_aids, axis=1)
    
        
    def get_feature_set1(self, PHASES_CUSTOM, PHASES_IDX=1):
        
        character_name = self.get_character_name(self.all_user_match_df)
        death_time_feat = self.feature_death_time(self.all_user_match_df, end_time = PHASES_CUSTOM[PHASES_IDX], start_time = PHASES_CUSTOM[PHASES_IDX-1])
        
        n_kill_feat = self.feature_kill_stats(self.all_user_match_df, 'kill', end_time = PHASES_CUSTOM[PHASES_IDX], start_time = PHASES_CUSTOM[PHASES_IDX-1])
        n_finish_feat = self.feature_kill_stats(self.all_user_match_df, 'finish', end_time = PHASES_CUSTOM[PHASES_IDX], start_time = PHASES_CUSTOM[PHASES_IDX-1])
        n_groggy_feat = self.feature_kill_stats(self.all_user_match_df, 'groggy', end_time = PHASES_CUSTOM[PHASES_IDX], start_time = PHASES_CUSTOM[PHASES_IDX-1])

        n_pickup_feat = self.feature_n_pickup(self.all_user_match_df, end_time = PHASES_CUSTOM[PHASES_IDX], start_time = PHASES_CUSTOM[PHASES_IDX-1])
        n_drop_feat = self.feature_n_drop(self.all_user_match_df, end_time = PHASES_CUSTOM[PHASES_IDX], start_time = PHASES_CUSTOM[PHASES_IDX-1])
        
        # Attack damage features
        dmg_attack_feat = self.feature_all_dmg(self.all_user_match_df, False, end_time = PHASES_CUSTOM[PHASES_IDX], start_time = PHASES_CUSTOM[PHASES_IDX-1])
        dmg_attack_feat.columns = ['n_dmg', 'sum_dmg']
        
        # 🎯 총기별 Attack damage features
        dmg_by_AR_feat = self.feature_dmg_by_weapon(self.all_user_match_df, 'AR', end_time = PHASES_CUSTOM[PHASES_IDX], start_time = PHASES_CUSTOM[PHASES_IDX-1])
        dmg_by_SR_feat = self.feature_dmg_by_weapon(self.all_user_match_df, 'SR', end_time = PHASES_CUSTOM[PHASES_IDX], start_time = PHASES_CUSTOM[PHASES_IDX-1])
        dmg_by_DMR_feat = self.feature_dmg_by_weapon(self.all_user_match_df, 'DMR', end_time = PHASES_CUSTOM[PHASES_IDX], start_time = PHASES_CUSTOM[PHASES_IDX-1])
        dmg_by_SMG_feat = self.feature_dmg_by_weapon(self.all_user_match_df, 'SMG', end_time = PHASES_CUSTOM[PHASES_IDX], start_time = PHASES_CUSTOM[PHASES_IDX-1])
        dmg_by_SHOTGUN_feat = self.feature_dmg_by_weapon(self.all_user_match_df, 'SHOTGUN', end_time = PHASES_CUSTOM[PHASES_IDX], start_time = PHASES_CUSTOM[PHASES_IDX-1])
        
        # Victim damage features (take damage)
        dmg_victim_feat = self.feature_all_dmg(self.all_user_match_df, True, end_time = PHASES_CUSTOM[PHASES_IDX], start_time = PHASES_CUSTOM[PHASES_IDX-1])
        dmg_victim_feat.columns = ['n_take_dmg', 'sum_take_dmg']
    
        
        out_df = pd.concat(
            [
                character_name, death_time_feat, n_kill_feat, n_finish_feat, n_groggy_feat,
                n_pickup_feat, n_drop_feat,
                dmg_attack_feat, dmg_by_AR_feat, dmg_by_SR_feat, dmg_by_DMR_feat, dmg_by_SMG_feat, dmg_by_SHOTGUN_feat,
                dmg_victim_feat
            ],
            axis=1
        )
        
        out_df.columns = [f'{name}' for name in [
                'character_name', 'death_time', 'n_kill', 'n_finish', 'n_groggy', 'n_pickup', 'n_drop'
            ] 
            + dmg_attack_feat.columns.tolist()
            + dmg_by_AR_feat.columns.tolist()
            + dmg_by_SR_feat.columns.tolist()
            + dmg_by_DMR_feat.columns.tolist()
            + dmg_by_SMG_feat.columns.tolist()
            + dmg_by_SHOTGUN_feat.columns.tolist()
            + dmg_victim_feat.columns.tolist()
        ]
            
        return out_df