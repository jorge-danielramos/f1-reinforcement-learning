import pandas as pd
import os
import duckdb

import fastf1

PARTITION_DIR = './data/f1_dataset'
CACHE_DIR = './cache'
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

def load_grand_prix(year:int, grand_prix:str, session:str='Race') -> pd.DataFrame:
    '''
    TODO
    - load gp data from cache if exists, otherwise fetch from API and save to cache
    - create hive partition by grand_prix, if already exists, load from there
    - return dataframe with all columns
    '''
    if not os.path.exists(PARTITION_DIR) or not os.listdir(PARTITION_DIR):
        partition_exists = False
    else:
        items = os.listdir(PARTITION_DIR)
        partition_exists = any("=" in item for item in items if os.path.isdir(os.path.join(PARTITION_DIR, item)))
        if partition_exists:
            all_laps = duckdb.sql(
                f"""
                SELECT * FROM
                read_parquet('{PARTITION_DIR}/**/*.parquet', hive_partitioning = true)
                WHERE GrandPrix = '{grand_prix}'
                AND Year = {year}
                """
            ).df()

            if len(all_laps) != 0:
                return all_laps
        else:
            raise RuntimeError("Attempted to create a partition, but the target directory is not empty and does not contain any partitions. Please clear the target directory and try again.")

    session = fastf1.get_session(year, grand_prix, session)
    session.load()

    all_laps = session.laps.copy().sort_values('Time')
    weather_data = session.weather_data.copy().sort_values('Time')
    track_status = session.track_status.copy().sort_values('Time').rename(columns={'Status': 'GlobalTrackStatus'})

    all_laps = pd.merge_asof(
        all_laps,
        weather_data[['Time', 'TrackTemp', 'AirTemp']],
        on='Time',
        direction='nearest'
    )
    all_laps = pd.merge_asof(
        all_laps,
        track_status[['Time', 'GlobalTrackStatus']],
        on='Time',
        direction='backward'
    )
    os.makedirs(PARTITION_DIR, exist_ok=True)

    duckdb.sql(
        f"""
        COPY (
            SELECT *, '{grand_prix}' AS GrandPrix, {year} AS Year FROM all_laps
            ORDER BY DriverNumber, LapNumber
            )
        TO '{PARTITION_DIR}'
        (FORMAT parquet, PARTITION_BY (GrandPrix, Year), OVERWRITE_OR_IGNORE)
        """
    )
    all_laps = duckdb.sql(
        f"""
        SELECT * FROM
        read_parquet('{PARTITION_DIR}/**/*.parquet', hive_partitioning = true)
        WHERE GrandPrix = '{grand_prix}'
        AND Year = {year}
        """
    ).df()# this fixes certain data type issues with the datetime columns
    
    return all_laps