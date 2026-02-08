"""
High RP Matches CSV 전처리 스크립트
matchid에서 year, month, day, hour를 추출하여 컬럼으로 추가
"""

import pandas as pd
import os

def extract_datetime_from_matchid(matchid):
    """
    matchid에서 year, month, day, hour 정보 추출
    
    matchid 예시: match.bro.competitive.pc-2018-37.steam.squad.as.2025.10.01.19.00811236-0c6f-4d9c-b720-94e649366267
    토큰 구조: match.bro.{league_type}.{season}.{platform}.{mode}.{region}.{year}.{month}.{day}.{hour}.{uuid}
    """
    try:
        tokens = matchid.split('.')
        year = tokens[7]
        month = tokens[8]
        day = tokens[9]
        hour = tokens[10]
        
        return year, month, day, hour
    except Exception as e:
        print(f"Error parsing matchid: {matchid}, Error: {e}")
        return None, None, None, None

def preprocess_high_rp_matches(input_csv_path, output_csv_path=None):
    """
    High RP matches CSV 파일을 읽어 year, month, day, hour 컬럼 추가
    
    Args:
        input_csv_path: 원본 CSV 파일 경로
        output_csv_path: 저장할 CSV 파일 경로 (None이면 _with_datetime.csv 접미사 추가)
    
    Returns:
        처리된 DataFrame
    """
    print(f"📖 Reading CSV file: {input_csv_path}")
    
    # CSV 파일 읽기
    df = pd.read_csv(input_csv_path)
    
    print(f"✅ Loaded {len(df)} rows")
    print(f"📊 Columns: {list(df.columns)}")
    
    # matchid 컬럼이 있는지 확인
    if 'matchid' not in df.columns:
        raise ValueError("CSV file must contain 'matchid' column")
    
    # year, month, day, hour 추출
    print(f"🔄 Extracting datetime information from matchid...")
    datetime_info = df['matchid'].apply(extract_datetime_from_matchid)
    
    # 새로운 컬럼 추가
    df['year'] = datetime_info.apply(lambda x: x[0])
    df['month'] = datetime_info.apply(lambda x: x[1])
    df['day'] = datetime_info.apply(lambda x: x[2])
    df['hour'] = datetime_info.apply(lambda x: x[3])
    
    # 파싱 실패한 행 확인
    failed_rows = df[df['year'].isna()]
    if len(failed_rows) > 0:
        print(f"⚠️  Warning: {len(failed_rows)} rows failed to parse")
        print(f"   Example failed matchid: {failed_rows.iloc[0]['matchid']}")
    
    # 파싱 성공한 행만 유지
    df_clean = df.dropna(subset=['year', 'month', 'day', 'hour'])
    
    print(f"✅ Successfully parsed {len(df_clean)} rows")
    print(f"📊 Date range:")
    print(f"   - Years: {sorted(df_clean['year'].unique())}")
    print(f"   - Months: {sorted(df_clean['month'].unique())}")
    
    # 날짜/시간별 통계
    date_hour_counts = df_clean.groupby(['year', 'month', 'day', 'hour']).size().reset_index(name='count')
    print(f"\n📈 Statistics:")
    print(f"   - Total unique date-hour combinations: {len(date_hour_counts)}")
    print(f"   - Total matches: {len(df_clean)}")
    print(f"   - Average matches per date-hour: {len(df_clean) / len(date_hour_counts):.2f}")
    
    # 출력 파일 경로 결정
    if output_csv_path is None:
        base_name = os.path.splitext(input_csv_path)[0]
        output_csv_path = f"{base_name}_with_datetime.csv"
    
    # CSV 저장
    print(f"\n💾 Saving to: {output_csv_path}")
    df_clean.to_csv(output_csv_path, index=False)
    print(f"✅ Saved successfully!")
    
    return df_clean

def create_matchid_lookup_dict(df):
    """
    날짜/시간별로 matchid를 그룹화한 딕셔너리 생성
    메모리 효율적인 조회를 위한 자료구조
    
    Args:
        df: year, month, day, hour, matchid 컬럼을 포함한 DataFrame
    
    Returns:
        {(year, month, day, hour): set(matchid1, matchid2, ...)}
    """
    print(f"\n🔨 Creating matchid lookup dictionary...")
    
    lookup_dict = {}
    
    for _, row in df.iterrows():
        key = (row['year'], row['month'], row['day'], row['hour'])
        if key not in lookup_dict:
            lookup_dict[key] = set()
        lookup_dict[key].add(row['matchid'])
    
    print(f"✅ Created lookup dictionary:")
    print(f"   - Unique date-hour keys: {len(lookup_dict)}")
    print(f"   - Total matches: {sum(len(v) for v in lookup_dict.values())}")
    
    return lookup_dict

def main():
    """메인 실행 함수"""
    
    # 파일 경로 설정 (Databricks 환경 기준)
    BASE_PATH = "/Volumes/main_dev/dld_ml_anticheat_test/anticheat_test_volume/pgc_wwcd/competitive_high_rp_match/"
    INPUT_CSV = f"{BASE_PATH}/high_rp_matches_as_squad_fpp_20250525_20251125.csv"
    OUTPUT_CSV = f"{BASE_PATH}/high_rp_matches_as_squad_fpp_20250525_20251125_with_datetime.csv"
    
    print("="*60)
    print("High RP Matches CSV 전처리")
    print("="*60)
    
    # CSV 전처리
    df = preprocess_high_rp_matches(INPUT_CSV, OUTPUT_CSV)
    
    # Lookup 딕셔너리 생성 (테스트용)
    lookup_dict = create_matchid_lookup_dict(df)
    
    # 샘플 출력
    print(f"\n📋 Sample lookup entries:")
    for i, (key, matchids) in enumerate(list(lookup_dict.items())[:3]):
        print(f"   {key}: {len(matchids)} matches")
        if i == 0:
            print(f"      Example matchid: {list(matchids)[0]}")
    
    print("\n✅ Preprocessing complete!")
    print(f"📁 Output file: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
