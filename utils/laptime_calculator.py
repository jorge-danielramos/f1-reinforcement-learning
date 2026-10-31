import numpy as np
from datetime import timedelta

from typing import (
    Dict,
    Any
)

from utils.track_status_calculator import get_red_flag_end_time

def calculate_laptime(
        baseline:timedelta,
        tyre_degradation:Dict[str, Any],
        pit_baselines:Dict[str, Any],
        compound:str,
        tyre_life:int,
        track_status:str,
        in_pit:bool,
        clock_time:timedelta,
        seed=None,
        track_status_rows=None
    ):
    np.random.seed(seed)
    track_status = str(track_status or '1')

    sc = '4' in track_status
    vsc = '6' in track_status or '7' in track_status

    if in_pit and '5' not in track_status: # pitting in red flag conditions costs no time
        time_in_pits = calculate_pit_time(pit_baselines, sc, vsc)
    else:
        time_in_pits = timedelta(0)
    
    compound_name = {1: 'SOFT', 2: 'MEDIUM', 3: 'HARD'}[compound]

    degradation_coefficients = tyre_degradation[compound_name]
    predicted_lap_time = np.polyval(degradation_coefficients, tyre_life)

    minimum = -0.5
    average = 0
    percentile99 = 3

    adj_mean = average - minimum
    adj_p99 = percentile99 - minimum
    a, b, c = 0.5, -2.326, np.log(adj_p99 / adj_mean)
    sigma = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
    mu = np.log(adj_mean) - (sigma**2 / 2)
    additive_noise = np.random.lognormal(mean=mu, sigma=sigma, size=1)[0]

    time_floor = baseline + 0.02 * tyre_life
    if tyre_life > 20:
        predicted_lap_time = max(predicted_lap_time, time_floor)

    lap_time = timedelta(seconds=predicted_lap_time) + time_in_pits + timedelta(seconds=additive_noise)

    red_delay = None
    if '5' in track_status:
        red_delay = get_red_flag_end_time(time=clock_time, track_status_rows=track_status_rows)

    if red_delay is None:
        delay = clock_time + lap_time
    elif red_delay < (clock_time + lap_time):
        delay = red_delay + lap_time
    else:
        delay = red_delay

    return lap_time, delay

def calculate_pit_time(pit_baselines:Dict[str, Any], sc:bool, vsc:bool, seed=None):
    minimum = -1
    average = 0
    percentile99 = 10

    adj_mean = average - minimum
    adj_p99 = percentile99 - minimum
    a, b, c = 0.5, -2.326, np.log(adj_p99 / adj_mean)
    sigma = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
    mu = np.log(adj_mean) - (sigma**2 / 2)
    additive_noise = np.random.lognormal(mean=mu, sigma=sigma, size=1)[0]
    if sc:
        baseline = pit_baselines['sc']
    elif vsc:
        baseline = pit_baselines['vsc']
    else:
        baseline = pit_baselines['normal']

    return timedelta(seconds=baseline + additive_noise)
