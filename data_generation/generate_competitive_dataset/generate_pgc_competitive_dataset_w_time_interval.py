# Databricks notebook source
# MAGIC %md
# MAGIC ### PGC WWCD Prediction - Competitive Squad-Fpp Dataset Generation
# MAGIC * Source: s3://prod-live-gamelog-service-new/gamelog/gamelog/bro/
# MAGIC * Saved Source: /Volumes/main_dev/dld_ml_anticheat_test/anticheat_test_volume/pgc_wwcd/pgc_match_log  
# MAGIC * Destination: /Volumes/main_dev/dld_ml_anticheat_test/anticheat_test_volume/pgc_wwcd/pgc_features/replay_samples_v2.1/
# MAGIC
# MAGIC #### Preprecessing steps
# MAGIC 1. Copy the source data to the local system.
# MAGIC 3. Parse the data and extract relevant features.
# MAGIC 4. Save the extracted features as features.csv.

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import sys, os

BASE = os.path.join(os.getcwd(), "..")
BASE = os.path.abspath(BASE)

if BASE not in sys.path:
    sys.path.insert(0, BASE)

print("BASE:", BASE)

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore", message=".*Mean of empty slice.*")
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")

import os
import itertools
import glob
import time
import pandas as pd
import numpy as np
import concurrent.futures
import traceback
import itertools
from datetime import datetime, timedelta, timezone

from feature_engineering.variables import MAP_NAME_TO_LEGACY
from feature_engineering.variable_utils import get_seasons
# from feature_engineering.utils import s3filecopy
from feature_engineering.parsing_match_log import MatchLogParser
from feature_engineering.extracting_pgc_features import FeatureExtractor, SquadFeatureExtractor

# COMMAND ----------

SOURCE_PATH_TEMP = 's3://prod-live-gamelog-service-new/gamelog/gamelog/bro/{year}/{month}/{day}/{hour}/{platform}/{league_type}/pc-2018-{season}/{mode}/{region}' # adding region 
TEMP_LOCAL_PATH = "/Volumes/main_dev/dld_ml_anticheat_test/anticheat_test_volume/pgc_wwcd/pgc_match_log/"

# Replay 1 sec sampled Squad-FPP 
DEST_PATH = "/Volumes/main_dev/dld_ml_anticheat_test/anticheat_test_volume/pgc_wwcd/pgc_features/replay_samples_v2.1/"

print(f"SOURCE_PATH_TEMP: {SOURCE_PATH_TEMP}")
print(f"TEMP_LOCAL_PATH: {TEMP_LOCAL_PATH}")
print(f"DEST_PATH: {DEST_PATH}")

# COMMAND ----------

def matchid2s3path(matchid):
    tokens = matchid.split('.')
    s3path = SOURCE_PATH_TEMP.format(
        year=tokens[7], month=tokens[8], day=tokens[9], hour=tokens[10], platform=tokens[4], league_type=tokens[2], season=tokens[3].split('-')[-1], region=tokens[6], mode=tokens[5], matchid=matchid.replace('match','gamelog')
    )
    return s3path

def check_s3_exist(s3path):
    try:
        res = dbutils.fs.ls(s3path)
    except:
        print(s3path)
        return f'no file' 
    
    if len(res) == 1:
        return True
    else:
        # raise Exception('more than one')
        return 'more than one file'

def s3filecopy(path, temp_path=TEMP_LOCAL_PATH):
    try:
        local_path = temp_path + path.split('s3://prod-live-gamelog-service-new/gamelog/gamelog/')[-1]
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if os.path.exists(local_path):
            return 'exists', local_path
        else:
            dbutils.fs.cp(path, local_path)
        return 'success', local_path
    
    except Exception as e:
        print(f"Error copying file from {path}: {e}")
        return 'error', None

def parallel_copy_to_local(match_json_list):
    st = time.time()
    all_status = []
    copy_ress = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
        futures = {executor.submit(s3filecopy, json_path, TEMP_LOCAL_PATH): json_path for json_path in match_json_list}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            json_path = futures[future]
            if i % 50 == 0:
                print(i, json_path, f'parsing ... done in {time.time()-st:.2f} seconds')
            try:
                status, copy_res = future.result()
                all_status.append(status)
                if copy_res is not None:
                    copy_ress.append(copy_res)
                    
            except Exception as e:
                print('error processing', json_path)
                traceback.print_exc() 
    
    return all_status, copy_ress

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load High RP Matches Filter

# COMMAND ----------

def load_high_rp_matches_filter(csv_path):
    """
    High RP matches CSV를 로드하여 날짜/시간별로 matchid를 그룹화한 딕셔너리 생성
    
    Args:
        csv_path: high_rp_matches CSV 파일 경로 (year, month, day, hour, matchid 컬럼 포함)
    
    Returns:
        lookup_dict: {(year, month, day, hour): set(matchid1, matchid2, ...)}
        all_matchids: 전체 matchid set (빠른 조회용)
    """
    print(f"\n{'='*60}")
    print(f"📖 Loading High RP Matches Filter")
    print(f"{'='*60}")
    print(f"📁 CSV Path: {csv_path}")
    
    try:
        # CSV 로드
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} high RP matches")
        
        # 필수 컬럼 확인
        required_cols = ['matchid', 'year', 'month', 'day', 'hour']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # 날짜/시간별 matchid 그룹화
        lookup_dict = {}
        all_matchids = set()
        
        for _, row in df.iterrows():
            # 키: (year, month, day, hour) - 모두 문자열로 통일
            key = (str(row['year']), str(row['month']).zfill(2), 
                   str(row['day']).zfill(2), str(row['hour']).zfill(2))
            
            if key not in lookup_dict:
                lookup_dict[key] = set()
            
            matchid = row['matchid']
            lookup_dict[key].add(matchid)
            all_matchids.add(matchid)
        
        print(f"\n📊 Filter Statistics:")
        print(f"   - Unique date-hour combinations: {len(lookup_dict)}")
        print(f"   - Total high RP matches: {len(all_matchids)}")
        print(f"   - Average matches per date-hour: {len(all_matchids) / len(lookup_dict):.2f}")
        
        # 샘플 출력
        print(f"\n📋 Sample entries:")
        for i, (key, matchids) in enumerate(list(lookup_dict.items())[:3]):
            print(f"   {key}: {len(matchids)} matches")
        
        print(f"\n✅ High RP Matches Filter loaded successfully!")
        print(f"{'='*60}\n")
        
        return lookup_dict, all_matchids
        
    except Exception as e:
        print(f"❌ Error loading High RP Matches Filter: {e}")
        print(f"   Filter will be DISABLED - all matches will be processed")
        return {}, set()

def extract_matchid_from_path(path):
    """
    파일 경로에서 matchid 추출
    
    Args:
        path: S3 또는 로컬 파일 경로
    
    Returns:
        matchid (str) 또는 None
    """
    try:
        # 파일명에서 matchid 추출
        # 예: gamelog.bro.competitive.pc-2018-37.steam.squad.as.2025.10.01.19.00811236-...json.gz
        filename = path.split('/')[-1]
        
        # gamelog를 match로 변경
        if filename.startswith('gamelog.'):
            matchid = filename.replace('gamelog.', 'match.').replace('.json.gz', '').replace('.json', '')
            return matchid
        
        return None
    except Exception as e:
        print(f"Error extracting matchid from path: {path}, Error: {e}")
        return None

# COMMAND ----------

# MAGIC %md
# MAGIC ### Time interval Sampling

# COMMAND ----------

# Setting sampling time interval
# e.g. 1초부터 1800초까지 1초마다 sampling
time_interval = 10
start_time = 1
end_time = 1800

# COMMAND ----------

def processing_match_log_task(path, idx, map_filter=None):
    try:
        import sys
        print(f"[{idx}] 🔄 START Processing: {path}", flush=True)
        sys.stdout.flush()
        
        # Step 1: Parse log
        print(f"[{idx}] Step 1: Parsing match log...", flush=True)
        log_parser = MatchLogParser(path, source='ded')
        player_df = log_parser.parse()
        print(f"[{idx}] ✅ Parsed: {player_df.shape}", flush=True)

        # Step 2: Extract map name
        print(f"[{idx}] Step 2: Extracting map name...", flush=True)
        if 'mapName' in player_df.columns:
            map_name = player_df['mapName'].iloc[0]
            print(f"[{idx}]    - Map from player_df.mapName: {map_name}", flush=True)
        else:
            # mapName 컬럼이 없으면 log_parser.match_df에서 추출
            map_name = log_parser.match_df[log_parser.match_df._t == 'logmatchdefinition'].dropna(axis=1, how='all').iloc[0]["submodeid"]
        
        print(f"[{idx}] 🗺️  Map: {map_name}", flush=True)

        # Step 3: Map filtering
        print(f"[{idx}] Step 3: Checking map filter...", flush=True)
        try:
            global q_map
            if q_map is not None and map_name not in q_map:
                print(f"[{idx}] ⏭️  SKIPPED (map filter): {map_name} not in allowed maps - {path}", flush=True)
                return None
        except NameError:
            print(f"[{idx}]    - q_map not defined, skipping filter", flush=True)
            pass

        # Step 4: Filter invalid players
        print(f"[{idx}] Step 4: Filtering invalid players...", flush=True)
        orig_len = len(player_df)
        print(f"[{idx}]    - Original player count: {orig_len}", flush=True)
        print(f"[{idx}]    - character_teamid NaN count: {player_df['character_teamid'].isna().sum()}", flush=True)
        
        player_df = player_df.dropna(subset=['character_teamid'])
        
        print(f"[{idx}]    - After filtering: {len(player_df)} players", flush=True)
        print(f"[{idx}]    - Removed: {orig_len - len(player_df)} players", flush=True)

        if len(player_df) == 0:
            print(f"[{idx}] ⚠️  RETURN None: No valid players after teamid filter", flush=True)
            return None

        # Step 5: Fill death_time
        print(f"[{idx}] Step 5: Processing death_time...", flush=True)
        death_time_nan = player_df['death_time'].isna().sum()
        player_df['death_time'] = player_df['death_time'].fillna(2000.0)

        # Step 6: Create squad_info
        print(f"[{idx}] Step 6: Creating squad info...", flush=True)
        squad_info = pd.DataFrame({
            'character_name': player_df['character_name'],
            'squad_number': player_df['character_teamid'].astype(int)
        })
        n_squads = squad_info['squad_number'].nunique()
        print(f"[{idx}]    - Players: {len(squad_info)}", flush=True)
        print(f"[{idx}]    - Squads: {n_squads}", flush=True)

        # Step 7: Create squad_labels
        print(f"[{idx}] Step 7: Creating squad labels...", flush=True)
        squad_labels = player_df.groupby('character_teamid').agg({
            'character_ranking': 'first',
            'death_time': 'max'
        }).reset_index()
        
        squad_labels.columns = ['squad_number', 'squad_ranking', 'squad_death_time']
        squad_labels['squad_number'] = squad_labels['squad_number'].astype(int)
        squad_labels['squad_ranking'] = squad_labels['squad_ranking'].astype(int)
        squad_labels['squad_win'] = (squad_labels['squad_ranking'] == 1).astype(int)
        print(f"[{idx}] 🏆 Squad labels: {len(squad_labels)} squads", flush=True)
        
        print(f"[{idx}]    - Squad labels shape: {squad_labels.shape}", flush=True)

        # Step 8: Extract match_id
        print(f"[{idx}] Step 8: Extracting match ID...", flush=True)
        match_id = player_df.index[0].split('@')[1] if '@' in player_df.index[0] else 'unknown_match'

        print(f"[{idx}]    - Match ID: {match_id}", flush=True)

        # Step 9: Create SquadFeatureExtractor
        print(f"[{idx}] Step 9: Creating SquadFeatureExtractor...", flush=True)
        
        squad_extractor = SquadFeatureExtractor(
            player_df, 
            squad_info, 
            zone_info=log_parser.zone_info
        )
        
        print(f"[{idx}]    - ✅ SquadFeatureExtractor created successfully", flush=True)

        # Step 10: Generate features (CRITICAL STEP)
        print(f"[{idx}] Step 10: Generating phase samples (CRITICAL)...", flush=True)
        print(f"[{idx}]      - match_id: {match_id}", flush=True)
        
        try:
            squad_dataset = squad_extractor.generate_phase_samples_w_time_interval(
                squad_labels=squad_labels,
                match_id=match_id,
                time_interval=time_interval, 
                start_time=start_time, 
                end_time=end_time
            )
            print(f"[{idx}]    - ✅ generate_phase_samples completed", flush=True)
            print(f"[{idx}]    - Result type: {type(squad_dataset)}", flush=True)
            
            if squad_dataset is None:
                print(f"[{idx}] ⚠️  RETURN None: generate_phase_samples returned None", flush=True)
                print(f"[{idx}]    - This likely means no valid samples were generated", flush=True)
                return None
            
            print(f"[{idx}]    - Result shape: {squad_dataset.shape}", flush=True)
            print(f"[{idx}]    - Result columns count: {len(squad_dataset.columns)}", flush=True)
            print(f"[{idx}]    - Sample columns: {list(squad_dataset.columns)[:10]}...", flush=True)
            
            if len(squad_dataset) == 0:
                print(f"[{idx}] ⚠️  RETURN None: Feature dataframe is empty", flush=True)
                return None
                
        except Exception as feat_error:
            print(f"[{idx}] ❌ EXCEPTION in generate_phase_samples:", flush=True)
            print(f"[{idx}]    - Error type: {type(feat_error).__name__}", flush=True)
            print(f"[{idx}]    - Error message: {feat_error}", flush=True)
            traceback.print_exc()
            return None
        
        # Step 11: Add metadata
        print(f"[{idx}] Step 11: Adding metadata...", flush=True)
        filename = path.split('/')[-1]
        mode = filename.split('.')[5] if len(filename.split('.')) > 5 else 'unknown'
        map_id = MAP_NAME_TO_LEGACY.get(map_name, None)
        
        squad_dataset = squad_dataset.copy()
        squad_dataset["mode"] = mode
        squad_dataset["map"] = map_id
        
        print(f"[{idx}]    - Mode: {mode}", flush=True)
        print(f"[{idx}]    - Map: {map_name}", flush=True)
        print(f"[{idx}]    - Final shape: {squad_dataset.shape}", flush=True)

        print(f"[{idx}] ✅ SUCCESS: Returning {len(squad_dataset)} rows", flush=True)
        print(f"{'='*80}\n", flush=True)
        
        return squad_dataset

    except Exception as e:
        import sys
        print(f"[{idx}] ❌ EXCEPTION: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        return None

# COMMAND ----------

# # Sample-test feature generation 
BASE_PATH = "/Volumes/main_dev/dld_ml_anticheat_test/anticheat_test_volume/pgc_wwcd/pgc_match_log/bro/2025/11/25/01/steam/competitive/pc-2018-38/squad-fpp/as/"

# feature v2.1 replay high rp matches 
match_files = ["gamelog.bro.competitive.pc-2018-38.steam.squad-fpp.as.2025.11.25.01.00a2cad8-c58c-442c-8364-262ebf456b7b.json.gz"]

for match_file in match_files:
    full_path = os.path.join(BASE_PATH, match_file)
    squad_dataset = processing_match_log_task(full_path, 0)
    folder_path = os.path.join(DEST_PATH)
    os.makedirs(folder_path, exist_ok=True)
    
    match_id = match_file.replace('.json.gz', '').split('.')[-1]
    filename = f"pgc_as_squad-fpp_features_labels_v2.1_{match_id}.csv"
    file_path = os.path.join(folder_path, filename)
    squad_dataset.to_csv(file_path, index=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setting feature generation option

# COMMAND ----------

q_platform = ["steam"] # "kakao" 
q_league_type = "competitive"
q_mode = ["squad-fpp"] # esport is limited to squad-fpp
q_device = "pc"
q_region = "as" # "sea"
q_map = ['Baltic_Main', 'Desert_Main', 'Tiger_Main', 'Neon_Main'] # ['erangel', 'miramar', 'taego', 'rondo'] # 

# 날짜 범위 설정
start_date = "2025-10-20"  # 시작 날짜
end_date = "2025-10-20"    # 종료 날짜

# COMMAND ----------

from datetime import datetime, timedelta

start_dt = datetime.strptime(start_date, "%Y-%m-%d")
end_dt = datetime.strptime(end_date, "%Y-%m-%d")

date_range = []
current_dt = start_dt
while current_dt <= end_dt:
    date_range.append(current_dt)
    current_dt += timedelta(days=1)

print(f"📅 처리할 날짜 범위: {start_date} ~ {end_date}")
print(f"📅 총 {len(date_range)}일 처리 예정")

# COMMAND ----------

# High ranking point matches filtering
HIGH_RP_CSV_PATH = "/Volumes/main_dev/dld_ml_anticheat_test/anticheat_test_volume/pgc_wwcd/competitive_high_rp_match/high_rp_matches_as_squad_fpp_20250525_20251125_with_datetime.csv" # as

# Initialize High RP Matches Filter
try:
    high_rp_lookup, high_rp_matchids = load_high_rp_matches_filter(HIGH_RP_CSV_PATH)
    print(f"✅ High RP Filter enabled: {len(high_rp_matchids)} matches loaded")
except Exception as e:
    print(f"⚠️  Warning: Failed to load High RP Filter - {e}")
    print(f"   Processing will continue WITHOUT filtering")
    high_rp_lookup = {}
    high_rp_matchids = set()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Multi-parallel Parsing

# COMMAND ----------

def parallel_parsing(match_json_list, map_filter=None):
    import concurrent.futures, time, traceback, pandas as pd
    import sys

    from collections import defaultdict

    st = time.time()
    all_squad_df = []
    success_count = 0
    none_count = 0
    error_count = 0

    # 통계 추적
    stats = {
        'success': 0,
        'none_returned': 0,
        'exception': 0,
        'none_reasons': defaultdict(int)  # None 반환 이유별 카운트
    }

    print(f"\n{'='*80}")
    print(f"🚀 Starting parallel parsing of {len(match_json_list)} files")
    print(f"   Workers: Max")
    print(f"   Map filter: {map_filter}")
    print(f"{'='*80}\n")
    sys.stdout.flush()

    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        futures = {executor.submit(processing_match_log_task, json_path, idx, map_filter): json_path
                 for idx, json_path in enumerate(match_json_list)}

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            json_path = futures[future]

            # 진행상황 출력
            if i % 10 == 0 or i == len(match_json_list) - 1:
                elapsed = time.time() - st
                avg_time = elapsed / (i + 1) if i > 0 else 0
                eta = avg_time * (len(match_json_list) - i - 1)
                
                print(f"\n📊 Progress: {i+1}/{len(match_json_list)} ({i+1/len(match_json_list)*100:.1f}%)")
                print(f"   ✅ Success: {stats['success']}")
                print(f"   ⚠️  None: {stats['none_returned']}")
                print(f"   ❌ Exception: {stats['exception']}")
                print(f"   ⏱️  Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")
                sys.stdout.flush()
            
            try:
                squad_df = future.result()
                
                if squad_df is not None and len(squad_df) > 0:
                    all_squad_df.append(squad_df)
                    success_count += 1
                else:
                    none_count += 1
                    if none_count <= 5:  # 처음 5개만 상세 로그
                        print(f"⚠️  File returned None: {json_path}")
            except Exception as e:
                error_count += 1
                print(f"❌ Exception processing {json_path}: {e}")
                traceback.print_exc()
                sys.stdout.flush()

    elapsed = time.time() - st
    print(f"\n{'='*60}")
    print(f"📊 Parallel Parsing Complete")
    print(f"{'='*60}")
    print(f"⏱️  Time elapsed: {elapsed:.2f}s")
    print(f"✅ Success: {success_count}/{len(match_json_list)}")
    print(f"⚠️  Returned None: {none_count}/{len(match_json_list)}")
    print(f"❌ Exceptions: {error_count}/{len(match_json_list)}")
    print(f"{'='*60}\n")

    sys.stdout.flush()

    if not all_squad_df:
        print("⚠️  No squad data parsed - returning None")
        return None

    # Squad 데이터 결합
    final_squad_df = pd.concat(all_squad_df, ignore_index=True)
    print(f"✅ Squad dataset shape: {final_squad_df.shape}")

    return final_squad_df

# COMMAND ----------

# 날짜/시간별 데이터 처리 함수
def process_single_date(target_date, target_hour, high_rp_lookup=None, high_rp_matchids=None):

    print(f"\n🔄 Processing date: {target_date} at {target_hour}:00")
    
    # Parsing date
    q_year, q_month, q_day = target_date.split("-")
    query_date = target_date
    q_season, _ = get_seasons(spark, query_date)
    
    print(f"📅 Date: {query_date}, Season: {q_season}, Hour: {target_hour}")
    
    # High RP 필터링: 날짜/시간 레벨 사전 체크
    use_filter = high_rp_lookup is not None and len(high_rp_lookup) > 0
    
    if use_filter:
        date_hour_key = (q_year, q_month, q_day, target_hour)
        
        if date_hour_key not in high_rp_lookup:
            print(f"⏭️  SKIPPED: No high RP matches for {date_hour_key}")
            return []
        else:
            expected_matchids = high_rp_lookup[date_hour_key]
            print(f"🎯 High RP Filter: {len(expected_matchids)} matches expected for this hour")
    else:
        print(f"⚠️  Filter DISABLED: Processing all matches")
        expected_matchids = None
    
    # S3 경로 생성 및 파일 수집
    queries = []
    try:
        for platform, mode in itertools.product(q_platform, q_mode):
            q_regions = SOURCE_PATH_TEMP.format(
            year=q_year, month=q_month, day=q_day, hour=target_hour, 
            platform=platform, mode=mode, season=q_season, 
            league_type=q_league_type, region=q_region
        )
            print(f"🔍 Checking: {q_regions}")
            try:
                for q in dbutils.fs.ls(q_regions):
                    paths = dbutils.fs.ls(q.path)
                    print(f"  📁 Found {len(paths)} files in {q.path}")
                    queries += [p.path for p in paths]
            except Exception as e:
                print(f"  ❌ No files found: {e}")
    except Exception as e:
        print(f"❌ Error collecting files: {e}")
    
    print(f"📊 Total files found (before filtering): {len(queries)}")
    
    # High RP 필터링: matchid 기반 파일 필터링
    if use_filter and expected_matchids is not None and len(queries) > 0:
        print(f"🔍 Filtering files by matchid...")
        
        filtered_queries = []
        matched_count = 0
        unmatched_count = 0
        
        for file_path in queries:
            matchid = extract_matchid_from_path(file_path)
            
            if matchid and matchid in expected_matchids:
                filtered_queries.append(file_path)
                matched_count += 1
            else:
                unmatched_count += 1
        
        print(f"✅ Filtering complete:")
        print(f"   - Matched: {matched_count} files")
        print(f"   - Filtered out: {unmatched_count} files")
        print(f"   - Expected: {len(expected_matchids)} matches")
        
        if matched_count < len(expected_matchids):
            print(f"⚠️  Warning: Found fewer files ({matched_count}) than expected ({len(expected_matchids)})")
        
        queries = filtered_queries
    
    print(f"📊 Total files to process: {len(queries)}")
    return queries

# COMMAND ----------

# 메인 루프: 날짜별 + 시간별 처리 (시간별 개별 저장)
print(f"\n🚀 Starting batch processing for {len(date_range)} days...")

# 전체 결과를 저장할 리스트
processed_dates_hours = []
failed_dates_hours = []
skipped_dates_hours = []  # 필터링으로 스킵된 시간대

# 각 날짜별로 처리
for i, current_date in enumerate(date_range):
    date_str = current_date.strftime("%Y-%m-%d")
    print(f"\n{'='*60}")
    print(f"📅 Processing {i+1}/{len(date_range)}: {date_str}")
    print(f"{'='*60}")
    
    # 해당 날짜의 모든 시간대 처리
    for hour in range(24):
        target_hour = f"{hour:02d}"
        print(f"\n🕐 Processing {date_str} at {target_hour}:00")
        
        try:
            # 1. 해당 시간대의 파일 수집 (필터 적용)
            queries = process_single_date(date_str, target_hour, high_rp_lookup, high_rp_matchids)
            
            if len(queries) == 0:
                print(f"⚠️ No files found for {date_str} at {target_hour}:00, skipping...")
                skipped_dates_hours.append(f"{date_str}_{target_hour}")
                continue
            
            # 2. 파일 복사
            print(f"📥 Copying {len(queries)} files...")
            all_status, save_files = parallel_copy_to_local(queries)
            print(f"📊 Copy status: {pd.Series(all_status).value_counts()}")
            
            if len(save_files) == 0:
                print(f"⚠️ No files copied for {date_str} at {target_hour}:00, skipping...")
                continue
            
            # 3. Feature 추출
            print(f"🔧 Extracting features from {len(save_files)} files...")
            feat_df = parallel_parsing(save_files, map_filter=q_map)
            
            if feat_df is not None and len(feat_df) > 0:
                print(f"✅ Successfully processed {date_str} at {target_hour}:00: {feat_df.shape}")
                
                # 4. 시간별로 즉시 저장
                year, month, day = date_str.split("-")
                # date_folder = f"{year}/{month}/{day}"
                folder_path = os.path.join(DEST_PATH)
                os.makedirs(folder_path, exist_ok=True)
                
                filename = f"pgc_as_squad-fpp_features_labels_{time_interval}sec_sampled_v2.1_{year}{month}{day}_{target_hour}.csv"
                file_path = os.path.join(folder_path, filename)
                feat_df.to_csv(file_path, index=False)
                
                print(f"💾 Saved {date_str} at {target_hour}:00: {file_path} ({len(feat_df)} samples)")
                
                # 성공 기록
                processed_dates_hours.append(f"{date_str}_{target_hour}")
            else:
                print(f"❌ Feature extraction failed for {date_str} at {target_hour}:00")
                failed_dates_hours.append(f"{date_str}_{target_hour}")
                
        except Exception as e:
            print(f"❌ Error processing {date_str} at {target_hour}:00: {e}")
            failed_dates_hours.append(f"{date_str}_{target_hour}")
            continue

print(f"\n📊 Batch Processing Summary:")
print(f"✅ Successfully processed: {len(processed_dates_hours)} date-hour combinations")
print(f"⏭️  Skipped (by filter): {len(skipped_dates_hours)} date-hour combinations")
print(f"❌ Failed: {len(failed_dates_hours)} date-hour combinations")
total_combinations = len(date_range) * 24
print(f"📈 Success rate: {len(processed_dates_hours)}/{total_combinations} ({len(processed_dates_hours)/total_combinations*100:.1f}%)")
if len(skipped_dates_hours) > 0:
    print(f"🎯 Filter efficiency: Skipped {len(skipped_dates_hours)}/{total_combinations} ({len(skipped_dates_hours)/total_combinations*100:.1f}%) time slots")


# 성공/실패 상세 정보
if len(processed_dates_hours) > 0:
    print(f"\n✅ Successfully processed date-hours:")
    for dh in processed_dates_hours[:10]:  # 처음 10개만 표시
        print(f"  - {dh}")
    if len(processed_dates_hours) > 10:
        print(f"  ... and {len(processed_dates_hours) - 10} more")

if len(failed_dates_hours) > 0: 
    print(f"\n❌ Failed date-hours:")
    for dh in failed_dates_hours[:10]:  # 처음 10개만 표시
        print(f"  - {dh}")
    if len(failed_dates_hours) > 10:
        print(f"  ... and {len(failed_dates_hours) - 10} more")