from datetime import timedelta
import numpy as np

import gymnasium as gym
from gymnasium import spaces

from utils.laptime_calculator import calculate_laptime
from utils.race_information import calculate_race_information
from utils.load_grand_prix import load_grand_prix

def build_track_status_rows(laps):
    rows = laps[['TrackStatus', 'GlobalTrackStatus', 'Time']].dropna(subset=['Time']).copy()
    rows = rows.sort_values('Time').drop_duplicates(subset=['TrackStatus', 'GlobalTrackStatus', 'Time'])
    return tuple(rows.itertuples(index=False, name=None))

class F1Env(gym.Env):
    def __init__(self, year:int, grand_prix:str, agent_type:str='bandit', seed=None):
        super(F1Env, self).__init__()
        '''
        TODO info needed from build_dataset:
            - number of laps
            - baseline lap time
            - track temperature
            - time multiplier function for each compound based on stint length and track temp
            - average pit time
            - average sc pit time
        '''
        self.seed = seed
        self.agent_type = agent_type

        gp_information = calculate_race_information(year, grand_prix)

        self.baseline_laptime = gp_information['baseline_laptime']
        self.pit_times = gp_information['pit_times']
        self.tyre_degradation = gp_information['tyre_degradation']
        self.total_laps = gp_information['total_laps']
        self.current_clock_time = gp_information['start_time']
        self.clock_time_start = gp_information['start_time']
        self.track_status_rows = gp_information.get('track_status_rows')
        if self.track_status_rows is None:
            self.track_status_rows = build_track_status_rows(load_grand_prix(year, grand_prix))

        self.lap_number = 0
        self.track_status = '1'
        self.tyre_life = 0
        self.compound = 0
        self.used_compounds = {}
        self.last_lap_time = timedelta(0)
        self.total_race_time = timedelta(0)

        self.action_space = spaces.Discrete(4) # 0: keep current compound, 1: switch to soft, 2: switch to medium, 3: switch to hard

        if agent_type == 'bandit':
            agent_shape = 2
        elif agent_type == 'actor_critic' or agent_type == 'dqn':
            agent_shape = 7
        else:
            raise ValueError('Invalid agent type.')

        self.observation_space = spaces.Dict({
            'state': spaces.Box(low=0, high=1, shape=(agent_shape,), dtype=np.float32),
            'action_mask': spaces.Box(low=0, high=1, shape=(4,), dtype=np.int8)
        })
        
    
    def reset(self, seed=None):
        '''
        TODO
            - reset the environment to the initial state
            - return the initial observation and info
        '''
        if seed is not None:
            self.seed = seed

        self.lap_number = 0
        self.track_status = '1'
        self.tyre_life = 0
        self.compound = 0
        self.used_compounds = {}
        self.last_lap_time = timedelta(0)
        self.total_race_time = timedelta(0)
        self.current_clock_time = self.clock_time_start

        info = {}
        return self._get_obs(), info

    def step(self, action:int):
        '''
        TODO
            - apply the action to the environment
            - calculate the reward based on the action and the current state
            - return the new observation, reward, done, and info
        '''
        self.track_status = self._time_to_track_status(self.current_clock_time)

        if self.lap_number == 0 and action == 0:
            # DNS for not leaving the pits
            return self._get_obs(), -5000, True, {}
        elif self.lap_number == 0:
            self.last_lap_time, delay = self._calculate_laptime(
                baseline=self.baseline_laptime,
                compound=action,
                tyre_degradation=self.tyre_degradation,
                pit_baselines=self.pit_times,
                tyre_life=self.tyre_life,
                track_status=self.track_status,
                in_pit=False,
                clock_time=self.current_clock_time,
                seed=self.seed,
                track_status_rows=self.track_status_rows
            )
            self.compound = action
            if action in self.used_compounds:
                self.used_compounds[action] += 1
            else:
                self.used_compounds[action] = 1
        elif action != 0:
            self.last_lap_time, delay = self._calculate_laptime(
                baseline=self.baseline_laptime,
                compound=self.compound,
                tyre_degradation=self.tyre_degradation,
                pit_baselines=self.pit_times,
                tyre_life=self.tyre_life,
                track_status=self.track_status,
                in_pit=True,
                clock_time=self.current_clock_time,
                seed=self.seed,
                track_status_rows=self.track_status_rows
            )
            self.compound = action
            if action in self.used_compounds:
                self.used_compounds[action] += 1
            else:
                self.used_compounds[action] = 1
            self.tyre_life = 0
        else:
            self.last_lap_time, delay = self._calculate_laptime(
                baseline=self.baseline_laptime,
                compound=self.compound,
                tyre_degradation=self.tyre_degradation,
                pit_baselines=self.pit_times,
                tyre_life=self.tyre_life,
                track_status=self.track_status,
                in_pit=False,
                clock_time=self.current_clock_time,
                seed=self.seed,
                track_status_rows=self.track_status_rows
            )
            self.tyre_life += 1

        self.current_clock_time = delay
        self.total_race_time += self.last_lap_time
        self.lap_number +=1

        terminated = self.lap_number >= self.total_laps
        reward = -self.last_lap_time.total_seconds()
        # Force at least two compounds
        if terminated and len(self.used_compounds.keys()) < 2:
            reward -= 5000 #DSQ

        return self._get_obs(), reward, terminated, {}

    def _get_action_mask(self):
        action_array = np.array([1, 1, 1, 1], dtype=np.int8)

        if self.lap_number == 0:
            action_array[0] = 0
        elif (self.lap_number == self.total_laps - 1) and (len(self.used_compounds.keys()) < 2):
            # case when an agent has only pitted for the same compound over the course of the race or not at all
            # the regulations oblige them to pit for a different compound
            action_array[0] = 0
            action_array[self.compound] = 0 # not allowed to pit for the same compound
        else:
            for compound, num_used in self.used_compounds.items():
                if num_used > 2:
                    action_array[compound] = 0 # 3 is the maximum allocation of a given set of tyres
        
        return action_array

    def _get_obs(self):
        if self.agent_type == 'bandit':
            return {
                "state": np.array([self.lap_number, self._last_lap_seconds()], dtype=np.float32),
                "action_mask": self._get_action_mask()
            }
        elif self.agent_type == 'actor_critic' or self.agent_type == 'dqn':
            track_status_int = self._track_status_int()
            if '5' in str(self.track_status) and '7' in str(self.track_status):
                track_status_int = 5 # red flag overrides vsc ending

            state = np.array([
                self.total_laps - self.lap_number,
                self.tyre_life,
                self.compound,
                1 if 1 in self.used_compounds else 0,
                1 if 2 in self.used_compounds else 0,
                1 if 3 in self.used_compounds else 0,
                track_status_int
            ], dtype=np.float32)
            return {
                "state": state,
                "action_mask": self._get_action_mask()
            }

    def _last_lap_seconds(self):
        return self.last_lap_time.total_seconds() if self.last_lap_time is not None else 0.0

    def _time_to_track_status(self, time):
        rows = self.track_status_rows
        results = sorted(rows, key=lambda row: abs(time - row[2]))[:2]

        if len(results) < 2:
            return str(results[0][0]) if results else '1'

        before = results[0]
        after = results[1]

        track_status_before = str(before[0])
        global_status_before = str(before[1])
        track_status_after = str(after[0])

        if track_status_before == track_status_after:
            return track_status_before
        if global_status_before in track_status_before:
            return track_status_before
        if global_status_before in track_status_after:
            return track_status_after

        distance_before = abs(time - before[2])
        distance_after = abs(time - after[2])
        return track_status_before if distance_before <= distance_after else track_status_after

    def _calculate_laptime(self, **kwargs):
        try:
            return calculate_laptime(**kwargs)
        except TypeError as exc:
            if "track_status_rows" not in str(exc):
                raise

            fallback_kwargs = kwargs.copy()
            fallback_kwargs.pop("track_status_rows", None)
            return calculate_laptime(**fallback_kwargs)

    def _track_status_int(self):
        track_status = str(self.track_status or '1')
        digits = [int(c) for c in track_status if c.isdigit()]
        if '5' in track_status and '7' in track_status:
            return 5 # red flag overrides vsc ending
        return max(digits) if digits else 1
