import pandas as pd
import numpy as np
from datetime import timedelta

from typing import (
    Dict,
    Any
)

from utils.load_grand_prix import load_grand_prix

def build_track_status_rows(laps):
    rows = laps[['TrackStatus', 'GlobalTrackStatus', 'Time']].dropna(subset=['Time']).copy()
    rows = rows.sort_values('Time').drop_duplicates(subset=['TrackStatus', 'GlobalTrackStatus', 'Time'])
    return tuple(rows.itertuples(index=False, name=None))

def calculate_race_information(year: int, grand_prix: str) -> Dict[str, Any]:
    '''
    Calculate all statistics needed for lap time simulation based on the whole grid.
    
    Note: driver_number parameter is accepted but not used (for backward compatibility).
    All calculations are performed on whole field data.
    
    Args:
        year: Race year
        grand_prix: Grand Prix name
        driver_number: Deprecated. Ignored. Included for backward compatibility.
    
    Returns:
        dict: Contains:
            - baseline_laptime: float (seconds)
            - pit_times: dict with keys 'normal', 'sc', 'vsc' and values as float (seconds)
            - tyre_degradation: dict mapping compound -> degradation_rate (seconds per lap)
    '''
    all_laps = load_grand_prix(year, grand_prix)
    
    total_laps = max(all_laps['LapNumber'])
    start_time = all_laps['Time'].min()
    baseline_laptime = calculate_baseline_laptime(all_laps)
    pit_times = calculate_pit_times(all_laps)
    tyre_degradation = calculate_tyre_degradation(all_laps)
    track_status_rows = build_track_status_rows(all_laps)
    
    return {
        'total_laps': total_laps,
        'baseline_laptime': baseline_laptime,
        'pit_times': pit_times,
        'tyre_degradation': tyre_degradation,
        'start_time': start_time,
        'track_status_rows': track_status_rows
    }


def calculate_baseline_laptime(laptimes: pd.DataFrame) -> float:
    '''
    Calculate baseline lap time from clean laps in normal track conditions.
    
    - Filters to normal track status ('1')
    - Removes first lap of race and first lap of each stint (tire warmup)
    - Removes statistical outliers using IQR method
    
    Args:
        laptimes: DataFrame with all drivers' lap data
    
    Returns:
        float: Baseline lap time in seconds
    '''
    df = laptimes.copy()
    
    # Convert LapTime to seconds if needed
    if 'LapTimeSeconds' not in df.columns:
        if df['LapTime'].dtype in ['timedelta64[ns]', 'timedelta64[us]']:
            df['LapTimeSeconds'] = df['LapTime'].dt.total_seconds()
        else:
            df['LapTimeSeconds'] = df['LapTime']
    
    # Filter to normal track status (1 = green flag)
    df = df[df['TrackStatus'] == '1'].copy()

    # Remove laps with invalid times
    df = df[df['LapTimeSeconds'].notna() & (df['LapTimeSeconds'] > 0)]
    
    # Remove first lap of race (too much traffic and fighting for posistions)
    df = df[df['LapNumber'] > 1]
    
    # Remove first lap of each stint (FreshTyre = True, tires warming up)
    if 'FreshTyre' in df.columns:
        df = df[df['FreshTyre'] != True]
    
    if len(df) == 0:
        raise ValueError("No valid lap times found to calculate baseline laptime.")
    
    # Remove outliers using IQR method
    Q1 = df['LapTimeSeconds'].quantile(0.25)
    Q3 = df['LapTimeSeconds'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['LapTimeSeconds'] >= lower_bound) & (df['LapTimeSeconds'] <= upper_bound)]
    
    if len(df) == 0:
        raise ValueError("No valid lap times found to calculate baseline laptime.")
    
    return df['LapTimeSeconds'].median()


def calculate_pit_times(laptimes: pd.DataFrame) -> Dict[str, float]:
    '''
    Calculate average pit stop times in normal and special conditions.
    
    First attempts to identify pit times using the pattern where PitInTime occurs in lap N
    and PitOutTime occurs in lap N+1. Falls back to calculating from same-lap pit times
    if the pattern is not found.
    
    Segments by track status (normal, safety car, virtual safety car).
    
    Args:
        laptimes: DataFrame with all drivers' pit data
    
    Returns:
        dict: Contains:
            - 'normal': pit time under green flag (seconds)
            - 'sc': pit time under safety car (seconds)
            - 'vsc': pit time under virtual safety car (seconds)
    '''
    df = laptimes.copy()
    
    # First, try to find pit times using the pattern: PitInTime in lap N, PitOutTime in lap N+1
    pit_times_list = []
    
    if 'LapNumber' in df.columns:
        df = df.sort_values(['DriverNumber', 'LapNumber'])
        pit_in_indices = df[df['PitInTime'].notna()].index
        
        for idx in pit_in_indices:
            pit_in_time = df.loc[idx, 'PitInTime']
            current_lap = df.loc[idx, 'LapNumber']
            driver_number = df.loc[idx, 'DriverNumber']
            current_track_status = df.loc[idx, 'TrackStatus']
            
            # Look for the next lap with PitOutTime
            next_lap_row = df[df['LapNumber'] == current_lap + 1]
            next_lap_row = next_lap_row[next_lap_row['DriverNumber'] == driver_number]
            if len(next_lap_row) > 0:
                next_lap_row = next_lap_row.iloc[0]
                if next_lap_row['PitOutTime'] > timedelta(seconds=0):
                    next_track_status = next_lap_row['TrackStatus']
                    
                    # Skip pit times under red flag conditions (TrackStatus = 5)
                    if '5' in str(current_track_status) or '5' in str(next_track_status):
                        continue
                    
                    pit_duration = (next_lap_row['PitOutTime'] - pit_in_time).total_seconds()
                    if pit_duration > 0:
                        pit_times_list.append({
                            'pit_time': pit_duration,
                            'track_status': next_track_status
                        })
    
    # If we found pit times using the pattern, use those
    if pit_times_list:
        pit_laps = pd.DataFrame(pit_times_list)
        pit_times = {}
        
        # Normal pit times
        normal_pits = pit_laps[pit_laps['track_status'].astype(str).str.contains('^[12]+$', na=False, regex=True)]
        if len(normal_pits) > 0:
            pit_times['normal'] = normal_pits['pit_time'].mean()
        else:
            pit_times['normal'] = 25.0 # Default pit time under normal conditions
        
        # Safety car pit times
        sc_pits = pit_laps[pit_laps['track_status'].astype(str).str.contains('4', na=False)]
        if len(sc_pits) > 0:
            pit_times['sc'] = sc_pits['pit_time'].mean()
        else:
            pit_times['sc'] = 18.0 # Default pit time under safety car conditions
        
        # Virtual safety car pit times (TrackStatus = 6 for deployed, 7 for ending)
        vsc_pits = pit_laps[pit_laps['track_status'].astype(str).str.contains('6|7', na=False)]
        if len(vsc_pits) > 0:
            pit_times['vsc'] = vsc_pits['pit_time'].mean()
        else:
            pit_times['vsc'] = 18.0 # Default pit time under virtual safety car conditions
        
        return pit_times
    
    # Fall back to the current calculation method: pit time from same lap
    df['PitTime'] = (df['PitOutTime'] - df['PitInTime']).dt.total_seconds()
    
    # Exclude pit times under red flag conditions (TrackStatus = 5)
    pit_laps = df[(df['PitTime'].notna()) & 
                  (df['PitTime'] > 0) & 
                  (~df['TrackStatus'].astype(str).str.contains('5', na=False))].copy()
    
    if len(pit_laps) == 0:
        # No pit data available, return default values
        return {
            'normal': 25.0,
            'sc': 18.0,
            'vsc': 18.0
        }
    
    pit_times = {}
    
    # Normal pit times (TrackStatus == '1')
    normal_pits = pit_laps[pit_laps['TrackStatus'].astype(str).str.contains('^[12]+$', na=False, regex=True)]
    if len(normal_pits) > 0:
        pit_times['normal'] = normal_pits['PitTime'].mean()
    else:
        pit_times['normal'] = 25.0
    
    # Safety car pit times (contains '4' in TrackStatus)
    sc_pits = pit_laps[pit_laps['TrackStatus'].astype(str).str.contains('4', na=False)]
    if len(sc_pits) > 0:
        pit_times['sc'] = sc_pits['PitTime'].mean()
    else:
        pit_times['sc'] = 18.0
    
    # Virtual safety car pit times (TrackStatus = 6 or 7)
    vsc_pits = pit_laps[pit_laps['TrackStatus'].astype(str).str.contains('6|7', na=False)]
    if len(vsc_pits) > 0:
        pit_times['vsc'] = vsc_pits['PitTime'].mean()
    else:
        pit_times['vsc'] = 18.0
    
    return pit_times

def calculate_tyre_degradation(laptimes: pd.DataFrame) -> Dict[str, np.ndarray]:
    '''
    Calculate tire degradation rates for each compound using degree 4 polynomial fits.
    
    For each compound:
    1. Fits a degree 4 polynomial to TyreLife vs LapTime for each driver individually
    2. Removes outlier laps before fitting and rejects outlier drivers (those with unusual coefficients)
    3. Averages the polynomial coefficients across all drivers
    
    Args:
        laptimes: DataFrame with all drivers' tire data
    
    Returns:
        dict: Maps compound (SOFT, MEDIUM, HARD, etc.) -> np.ndarray of polynomial coefficients
              Coefficients are for: laptime = c[0] + c[1]*TyreLife + c[2]*TyreLife^2 + c[3]*TyreLife^3 + c[4]*TyreLife^4
    '''
    df = laptimes.copy()
    
    # Convert LapTime to seconds if needed
    if 'LapTimeSeconds' not in df.columns:
        if df['LapTime'].dtype in ['timedelta64[ns]', 'timedelta64[us]']:
            df['LapTimeSeconds'] = df['LapTime'].dt.total_seconds()
        else:
            df['LapTimeSeconds'] = df['LapTime']
    
    # Filter to valid laps with normal track status
    df = df[(df['TrackStatus'] == '1') & 
            (df['LapTimeSeconds'].notna()) & 
            (df['TyreLife'].notna()) &
            (df['Compound'].notna()) &
            (df['DriverNumber'].notna())].copy()
    
    if len(df) == 0:
        # If no valid data, return defaults
        return {
            'SOFT': np.array([222.0, -12996.0, 262329.0, -1958914.0, 81970638.0]),
            'MEDIUM': np.array([-2.59, -722.0, 39838.0, -589377.0, 81096883.0]),
            'HARD': np.array([20.0, -999.0, 23337.0, -285014.0, 79755430.0]),
            'INTERMEDIATE': np.array([-30.0, 5995.0, -210255.0, 1461585.0, 99531547.0]),
            'WET': np.array([1993.0, -71364.0, 938507.0, -5647158.0, 111584335.0])
        }
    
    degradation_rates = {}
    
    for compound in df['Compound'].unique():
        compound_data = df[df['Compound'] == compound]
        
        # Collect polynomial coefficients from each driver
        driver_coefficients = []
        
        for driver_number in compound_data['DriverNumber'].unique():
            driver_compound_laps = compound_data[compound_data['DriverNumber'] == driver_number]
            
            if len(driver_compound_laps) < 4:  # Need at least 4 points for degree 4 poly
                continue
            
            X = driver_compound_laps['TyreLife'].values.astype(float)
            y = driver_compound_laps['LapTimeSeconds'].values.astype(float)
            
            # Remove outliers using IQR method before fitting
            Q1 = np.percentile(y, 25)
            Q3 = np.percentile(y, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            mask = (y >= lower_bound) & (y <= upper_bound)
            X_clean = X[mask]
            y_clean = y[mask]
            
            if len(X_clean) < 4:  # Need at least 4 points after outlier removal
                continue
            
            # Fit degree 4 polynomial
            coeffs = np.polyfit(X_clean, y_clean, 4)
            driver_coefficients.append(coeffs)
        
        if len(driver_coefficients) == 0:
            # No valid data for this compound, skip
            continue
        
        # Convert to array for easier manipulation
        coefficients_array = np.array(driver_coefficients)
        
        # Remove outlier drivers using IQR method on each coefficient dimension
        # Only keep drivers whose coefficients are close to the median across all dimensions
        valid_drivers = np.ones(len(driver_coefficients), dtype=bool)
        
        for coeff_idx in range(5):
            coeff_values = coefficients_array[:, coeff_idx]
            Q1 = np.percentile(coeff_values, 25)
            Q3 = np.percentile(coeff_values, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            valid_drivers &= (coeff_values >= lower_bound) & (coeff_values <= upper_bound)
        
        # Average the valid driver coefficients
        valid_coeffs = coefficients_array[valid_drivers]
        if len(valid_coeffs) == 0:
            # If all drivers are outliers, use all of them
            valid_coeffs = coefficients_array
        
        avg_coeffs = np.mean(valid_coeffs, axis=0)
        degradation_rates[compound] = avg_coeffs
    
    # Fill in any missing compounds with defaults
    default_degradation = {
        'SOFT': np.array([222.0, -12996.0, 262329.0, -1958914.0, 81970638.0]),
        'MEDIUM': np.array([-2.59, -722.0, 39838.0, -589377.0, 81096883.0]),
        'HARD': np.array([20.0, -999.0, 23337.0, -285014.0, 79755430.0]),
        'INTERMEDIATE': np.array([-30.0, 5995.0, -210255.0, 1461585.0, 99531547.0]),
        'WET': np.array([1993.0, -71364.0, 938507.0, -5647158.0, 111584335.0])
    }
    for compound, default_coeffs in default_degradation.items():
        if compound not in degradation_rates:
            degradation_rates[compound] = default_coeffs
    
    return degradation_rates
