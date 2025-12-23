import hashlib
import random
from pathlib import Path
from datetime import datetime as dt
from datetime import timedelta

import pandas as pd
import numpy as np
import torch

from gymnasium import Env
from gymnasium import spaces

import yfinance as yf
ticker = "ANA.AQ"

from constants import *

# look into Databento

class AssetEnvironment(Env):
    def __init__(self,
                 symbols: list[str],
                 start_date: dt = dt(year=2025, month=12, day=20) - timedelta(days=50),
                 end_date: dt = dt(year=2025, month=12, day=20),
                 interval: str = '30m',
                 input_period_len: int = 50,
                 max_steps: int = 50,
                 commission_rate: float = 0.0025 # 0.25%
                ):
        
        super().__init__()
        # consts
        self._input_period_len_const = input_period_len
        self._symbols_const = symbols
        self._max_steps_const = max_steps
        self._commission_rate_const = commission_rate
        # init values
        self._portfolio: torch.Tensor = torch.zeros(1).float().to(DEVICE)
        self._set_initial_portfolio()

        # for sampling
        self._raw_data: pd.DataFrame = self.__init_get_raw_data(symbols, start_date, end_date, interval)

        # for calculating reward
        self._portfolio = np.zeros(shape=len(self._symbols_const)+1, dtype='float32')
        self._portfolio[0] = 1.0

        # for tracking obs location
        self._obs_index: list[int] = [0]
        self._initial_obs_index: list[int] = [0]
        self._step_number = 0

        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(symbols) + 1,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(input_period_len, 3, len(symbols)),
            dtype=np.float32
        )

    def __init_get_raw_data(self,
                            symbols: list[str],
                            start_date: dt, 
                            end_date: dt,
                            interval: str) -> pd.DataFrame:
        """
        Download the historic stock data
        """

        # todo move this to device here and slice from the device rather than the df in the rest of the environment
        str_to_hash = str((*symbols, start_date, end_date, interval))
        input_hash = hashlib.sha224(str_to_hash.encode()).hexdigest()
        raw_data_file_path = Path(f'asset_env_raw_data_{input_hash}.feather')
        if raw_data_file_path.exists():
            print(f'Using saved data {raw_data_file_path}')
            raw_data = pd.read_feather(raw_data_file_path)
        else:
            raw_data = yf.download(symbols, start=start_date, end=end_date, interval=interval)
            assert isinstance(raw_data, pd.DataFrame)
            if isinstance(raw_data.columns, pd.MultiIndex):
                raw_data.columns.names = ['Field', 'Ticker']
            else:
                raise ValueError("Expected MultiIndex columns from yfinance")

            raw_data.to_feather(raw_data_file_path)
        return raw_data
    
    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool, bool, dict]:
        """Move to the next datapoint in the stock data, update the portfolio stored, calculate the 
        rewards which is the return on the previous portfolio and minus the cost of changing the portfolio"""
        assert torch.isclose(action.sum(dim=-1), torch.tensor(1.0), atol=1e-4).all()
        
        new_portfolio = action

        obs = self._get_subsample()
        reward = self._get_reward(new_portfolio)
        self._portfolio = new_portfolio
        info = self._get_info()
        
        self._obs_index = [i+1 for i in self._obs_index]
        self._step_number += 1
        terminated = self._step_number >= self._max_steps_const
        truncated = False
        return obs, reward, terminated, truncated, info
    
    def _get_reward(self, new_portfolio: torch.Tensor) -> torch.Tensor:
        """The rate of return for the time period"""
        mu = self._get_transaction_remainder_factor(new_portfolio)
        y = self._get_price_relative_vector()
        reward = torch.log((1+mu) * (y * new_portfolio).sum(dim=1))
        return reward / self._max_steps_const
    
    def _get_transaction_remainder_factor(self, new_portfolio: torch.Tensor):
        """How much it costs to transition from one portfolio to another portfolio"""
        # TODO this comes from an iterative eq, this is just the initial value
        #   implement the rest
        value = self._commission_rate_const * torch.sum(torch.abs(new_portfolio - self._portfolio))
        assert 0 < value <=1
        return value

    def _get_price_relative_vector(self) -> torch.Tensor:
        """Data provided is close, high and low, normalise this data wrt the final close"""
        assert all([i > 0 for i in self._obs_index])
        assert all([i < len(self._raw_data) for i in self._obs_index]) 

        vt = self._raw_data.iloc[self._obs_index]['Close']
        vtm1 = self._raw_data.iloc[[i-1 for i in self._obs_index]]['Close']

        yt = vt.values / vtm1.values

        return torch.concat([
            torch.ones([len(self._obs_index),1]), 
            torch.from_numpy(yt.astype('float32'))
            ], dim=-1).float().to(DEVICE)
    
    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[torch.Tensor, dict]:
        """Reset portfolio and select new starting observation time
        
        Returns
        -------
        obs: Tensor - shape (Batch, History, Features, Symbols)
        """
        self._step_number = 0

        batch_size = 1
        if isinstance(options, dict):
            batch_size =  options.get('batch_size', 1)
        
        if isinstance(seed, int):
            super().reset(seed=seed)
        
        self._set_initial_portfolio() # rest to cash only

        # get indices that have at least self._input_period_len_const entries behind them
        self._obs_index = [i + self._input_period_len_const for i in 
                           random.sample(range(
                               len(self._raw_data) - self._input_period_len_const - self._max_steps_const) # cannot pick so early that we dont have a history, nor late we dont have an episode 
                               , batch_size)]
        self._initial_obs_index = self._obs_index # to track how many steps have been taken
        obs = self._get_subsample() # get the data ending at _obs_index
        info = self._get_info() # get the info dict

        return obs, info
    
    def _set_initial_portfolio(self):
        """Cash only"""
        self._portfolio = torch.zeros(len(self._symbols_const)+1).float().to(DEVICE)
        self._portfolio[0] = 1.0

    def _get_subsample(self) -> torch.Tensor:
        """Get the n rows before the observation date
        
        subsamples shape = (Batch, History, Features, Symbols)
        """
        assert len(self._obs_index) > 0

        sub_samples = []
        for i in range(len(self._obs_index)):
            obs_index = self._obs_index[i]
            sub_sample = self._raw_data.iloc[obs_index - self._input_period_len_const: obs_index].drop(['Open', 'Volume'], axis=1).copy()
            
            assert len(sub_sample) == self._input_period_len_const #sanity check
            
            final_close = sub_sample['Close'].iloc[-1]
            for ticker in self._symbols_const:
                cols_to_norm = (sub_sample.columns.get_level_values('Ticker') == ticker)
                if not isinstance(sub_sample, pd.DataFrame):
                    a=1
                sub_sample.loc[:, cols_to_norm] /= final_close[ticker]

            sub_sample = sub_sample.values.reshape(len(sub_sample), 3, len(self._symbols_const))
            sub_samples.append(sub_sample)
        
        all_sub_samples = np.stack(sub_samples)
        return torch.from_numpy(all_sub_samples).float().to(DEVICE)
    
    def render(self, mode: str = 'human'):
        raise NotImplementedError
    
    def _get_info(self) -> dict:
        return {}
