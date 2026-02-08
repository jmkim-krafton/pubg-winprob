"""
PGC JSON → match_df 어댑터

load_match_df_from_s3()와 동일한 로직으로 PGC JSON을 match_df로 변환합니다.
"""

import json
import gzip
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Set
import itertools


# Train feature 생성에 필요한 log type 목록 (STEP1_LOG_TYPES_ANALYSIS.md 참조)
REQUIRED_LOG_TYPES = {
    # match
    'logmatchdefinition',
    'logmatchstart',
    'logmatchend',
    'logplayercreate',
    'logplayermatchuserinfo',
    
    # kill/damage/heal/revive
    'logplayerkillv2',
    'logplayermakegroggy',
    'logplayerrevive',
    'logplayertakedamage',
    'logheal',
    
    # position
    'logplayerposition',
    'loggamestateperiodic',
    
    # vehicle
    'logvehicleride',
    'logvehicleleave',
    'logvehicledamage',
    'logvehicledestroy',
    
    # item pickup/drop/put
    'logitempickup',
    'logitemdrop',
    'logitempickupfromcarepackage',
    'logitempickupfromitempackage',
    'logitempickupfromlootbox',
    'logitempickupfromcustompackage',
    'logitempickupfromrandombox',
    'logitempickupfromvehicletrunk',
    'logitemputtovehicletrunk',
    
    # item attach/detach/equip/unequip/use
    'logitemattach',
    'logitemdetach',
    'logitemequip',
    'logitemunequip',
    'logitemuse',
    
    # Throwable
    'logplayerusethrowable',
    
    # bluechip revival
    'logrevivalaircraft'
}



def read_pgc_json(file_path: str) -> List[Dict]:
    """
    PGC JSON 파일을 읽습니다. (.gz 파일 및 JSONL 형식 지원)
    
    Args:
        file_path: JSON 파일 경로
        
    Returns:
        List[Dict]: JSON 이벤트 리스트
    """
    data = []
    
    # .gz 파일인 경우
    if file_path.endswith('.gz'):
        with gzip.open(file_path, 'rt', encoding='utf-8-sig') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                # JSONL 형식 (각 줄이 JSON 객체)
                f.seek(0)
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
    else:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                # JSONL 형식
                f.seek(0)
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
    
    return data


def load_match_df_from_postmatch(
    path: str, 
    use_log_types: Set[str] = None,
    use_cols: List[str] = None,
    return_df: bool = False
) -> Tuple[pd.DataFrame, str, int]:
    """
    PGC JSON 파일을 match_df로 변환합니다.
    
    load_match_df_from_s3()와 동일한 로직을 따릅니다:
    1. JSON 파일 읽기
    2. log type 필터링
    3. DataFrame 생성 (중첩 구조 유지!)
    4. 컬럼명 lowercase 변환
    5. _t 필드 lowercase 변환
    6. valid_columns 선택
    7. Match ID 추출
    
    Args:
        path: PGC JSON 파일 경로
        use_log_types: 사용할 log type 집합 (None이면 REQUIRED_LOG_TYPES 사용)
        use_cols: 사용할 컬럼 리스트 (None이면 모든 컬럼)
        return_df: DataFrame 반환 여부
        
    Returns:
        tuple: (DataFrame, MatchId, N) if return_df else (MatchId, N)
    """
    # 1. JSON 파일 읽기
    data = read_pgc_json(path)
    
    # 2. log type 필터링
    if use_log_types is None:
        use_log_types = REQUIRED_LOG_TYPES
    
    log_data = []
    for json_d in data:
        if json_d['_T'].lower() in use_log_types:
            log_data.append(json_d)
    
    # 3. DataFrame 생성 (중첩 구조 유지!)
    ded_df = pd.DataFrame(log_data)
    
    # 4. 컬럼명 lowercase 변환
    ded_df.columns = [col.lower() for col in ded_df.columns]
    
    # 5. valid_columns 선택
    if use_cols:
        valid_columns = ['_t', '_d'] + [col for col in use_cols if col in ded_df.columns]
    else:
        valid_columns = ded_df.columns.tolist()
    
    # 6. _t 필드 lowercase 변환
    print(ded_df)
    ded_df['_t'] = ded_df['_t'].apply(lambda x: x.lower())
        
    # 7. DataFrame 필터링
    df = ded_df[valid_columns]
    
    # 8. Match ID 추출
    N = len(df)
    
    # LogMatchDefinition에서 추출 시도
    LogMatchDefinition = df[df['_t'] == 'logmatchdefinition']
    if len(LogMatchDefinition) > 0 and 'matchid' in LogMatchDefinition.columns:
        LogMatchDefinition = LogMatchDefinition.dropna(axis=1, how='all')
        MatchId = LogMatchDefinition.iloc[0]['matchid']
    else:
        # LogMatchDefinition이 없으면 파일명에서 추출
        MatchId = Path(path).stem.split('.')[0] if '.' in Path(path).stem else Path(path).stem
        print(f"⚠️ LogMatchDefinition 없음. 파일명에서 Match ID 추출: {MatchId}")
    
    # 9. PGC 호환성: LogMatchEnd.common에 mapName 추가
    log_match_start_mask = df['_t'] == 'logmatchstart'
    log_match_end_mask = df['_t'] == 'logmatchend'
    
    if log_match_start_mask.any() and log_match_end_mask.any():
        # LogMatchStart에서 mapName 추출
        map_name = df[log_match_start_mask].iloc[0].get('mapname', 'unknown')
        
        # LogMatchEnd의 common 필드에 mapName 추가
        for idx in df[log_match_end_mask].index:
            if 'common' in df.columns and pd.notna(df.at[idx, 'common']):
                common_dict = df.at[idx, 'common']
                if isinstance(common_dict, dict):
                    # mapName이 없으면 추가
                    if 'mapName' not in common_dict:
                        common_dict['mapName'] = map_name
                        df.at[idx, 'common'] = common_dict
                        print(f"✅ LogMatchEnd.common에 mapName 추가: {map_name}")
    
    # 10. 반환
    if return_df:
        return df.copy(), MatchId, N
    else:
        return MatchId, N