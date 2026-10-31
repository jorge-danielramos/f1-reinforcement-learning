import duckdb
from datetime import timedelta
from functools import lru_cache

TRACK_STATUS_PATH = './data/f1_dataset'

@lru_cache(maxsize=None)
def _load_track_status_rows(track_status_path:str=TRACK_STATUS_PATH):
    query = f"""
    SELECT 
        TrackStatus,
        GlobalTrackStatus,
        Time
    FROM read_parquet('{track_status_path}/**/*.parquet', hive_partitioning = true)
    ORDER BY Time ASC
    """

    return tuple(duckdb.sql(query).fetchall())

def build_track_status_rows(laps):
    rows = laps[['TrackStatus', 'GlobalTrackStatus', 'Time']].dropna(subset=['Time']).copy()
    rows = rows.sort_values('Time').drop_duplicates(subset=['TrackStatus', 'GlobalTrackStatus', 'Time'])
    return tuple(rows.itertuples(index=False, name=None))

def _time_distance(reference_time:timedelta, candidate_time:timedelta):
    return abs(reference_time - candidate_time)

def time_to_track_status(time:timedelta, track_status_path:str=TRACK_STATUS_PATH, track_status_rows=None):
    """
    Finds the track status at a given time by querying the closest timestamps before and after.
    
    Args:
        time: The timedelta to find track status for
        track_status_path: Path to the parquet dataset
        
    Returns:
        The appropriate track status string based on the comparison logic
    """
    
    rows = track_status_rows if track_status_rows is not None else _load_track_status_rows(track_status_path)
    results = sorted(rows, key=lambda row: _time_distance(time, row[2]))[:2]
    
    if len(results) < 2:
        # Handle edge case: if we can't find both before and after, return closest
        return str(results[0][0]) if results else None
    
    # Extract the two rows (before and after)
    before = results[0]
    after = results[1]
    
    track_status_before = str(before[0])
    global_status_before = str(before[1])
    track_status_after = str(after[0])
    
    # If both TrackStatus values are identical
    if track_status_before == track_status_after:
        return track_status_before
    
    # If they differ, check if GlobalTrackStatus from first row is in either TrackStatus
    if global_status_before in track_status_before:
        return track_status_before
    
    if global_status_before in track_status_after:
        return track_status_after
    
    # Return closest in time to the input time parameter
    time_before = before[2]
    time_after = after[2]
    
    distance_before = abs(time - time_before)
    distance_after = abs(time - time_after)
    
    return track_status_before if distance_before <= distance_after else track_status_after

def get_red_flag_end_time(time:timedelta, track_status_path:str=TRACK_STATUS_PATH, track_status_rows=None):
    """
    Finds the end time of the red flag period that encompasses the given time.
    
    Args:
        time: The timedelta to find red flag end time for
        track_status_path: Path to the parquet dataset
        
    Returns:
        The end time of the red flag period if found, otherwise None
    """
    
    rows = track_status_rows if track_status_rows is not None else _load_track_status_rows(track_status_path)
    green_flag_times = [
        row[2]
        for row in rows
        if '5' not in str(row[0]) and row[2] >= time
    ]
    
    return min(green_flag_times) if green_flag_times else None
