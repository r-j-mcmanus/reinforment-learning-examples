import hashlib
from pathlib import Path
from datetime import datetime as dt

import pandas as pd
import numpy as np
import torch

from gymnasium import Env
from gymnasium import spaces

import yfinance as yf

class AssetEnvironment(Env):
    def __init__(self,
                 symbols: list[str],
                 start_date: dt = dt(year=2025, month=11, day=1),
                 end_date: dt = dt(year=2025, month=12, day=1),
                 interval: str = '30m',
                 input_period_len: int = 50,
                 max_steps: int = 50,
                 commission_rate: float = 0.25
                ):
        
        super().__init__()
        # consts
        self._input_period_len_const = input_period_len
        self._symbols_const = symbols
        self._max_steps_const = max_steps
        self._commission_rate_const = commission_rate
        # init values
        self._portfolio: torch.Tensor = torch.zeros(1)
        self._set_initial_portfolio()

        # for sampling
        self._raw_data: pd.DataFrame = self.__init_get_raw_data(symbols, start_date, end_date, interval)

        # for calculating reward
        self._portfolio = np.zeros(shape=len(self._symbols_const)+1, dtype='float32')
        self._portfolio[0] = 1.0

        # for tracking obs location
        self._obs_index: int = 0
        self._initial_obs_index: int = 0

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
        str_to_hash = str((*symbols, start_date, end_date, interval))
        input_hash = hashlib.sha224(str_to_hash.encode()).hexdigest()
        raw_data_file_path = Path(f'asset_env_raw_data_{input_hash}.feather')
        if raw_data_file_path.exists():
            raw_data = pd.read_feather(raw_data_file_path)
        else:
            raw_data = yf.download(symbols, start=start_date, end=end_date, interval=interval)
            assert isinstance(raw_data, pd.DataFrame)
            raw_data.to_feather(raw_data_file_path)
        return raw_data
    
    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool, bool, dict]:
        assert np.isclose(sum(action.detach()), 1)
        
        new_portfolio = action

        self._obs_index += 1
        obs = self._get_subsample()
        reward = self._get_reward(new_portfolio)
        self._portfolio = new_portfolio
        info = self._get_info()
        terminated = (
            (self._obs_index >= (len(self._raw_data) - 1)) or
            ((self._obs_index - self._initial_obs_index) > self._max_steps_const)
        )
        truncated = False
        return obs, reward, terminated, truncated, info
    
    def _get_reward(self, new_portfolio: torch.Tensor) -> torch.Tensor:
        mu = self._get_transaction_remainder_factor(new_portfolio)
        y = self._get_price_relative_vector()
        reward = torch.log(mu * torch.dot(y, new_portfolio))
        return reward / self._max_steps_const
    
    def _get_transaction_remainder_factor(self, new_portfolio: torch.Tensor):
        # TODO this comes from an iterative eq, this is just the initial value
        #   implement the rest
        return self._commission_rate_const * torch.sum(torch.abs(new_portfolio - self._portfolio))
    
    def _get_price_relative_vector(self) -> torch.Tensor:
        vt = self._raw_data.iloc[self._obs_index]['Close']
        vtm1 = self._raw_data.iloc[self._obs_index - 1]['Close']

        yt = vt / vtm1

        return torch.concat([
            torch.ones(1), 
            torch.from_numpy(yt.to_numpy().astype('float32'))
            ])
    
    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[torch.Tensor, dict]:
        if isinstance(seed, int):
            self._np_random_seed = seed
        
        self._set_initial_portfolio()

        self._obs_index = np.random.randint(self._input_period_len_const, len(self._raw_data))
        self._initial_obs_index = self._obs_index
        obs = self._get_subsample()
        info = self._get_info()

        return obs, info
    
    def _set_initial_portfolio(self):
        self._portfolio = np.zeros(shape=len(self._symbols_const)+1, dtype='float32')
        self._portfolio[0] = 1.0
        self._portfolio: torch.Tensor = torch.from_numpy(self._portfolio)

    def _get_subsample(self) -> torch.Tensor:
        sub_sample = self._raw_data.iloc[self._obs_index - self._input_period_len_const: self._obs_index].drop(['Open', 'Volume'], axis=1)
        final_close = sub_sample['Close'].iloc[-1]
        for ticker in self._symbols_const:
            cols_to_norm = (sub_sample.columns.get_level_values('Ticker') == ticker)
            sub_sample.loc[:, cols_to_norm] /= final_close[ticker]

        sub_sample = sub_sample.values.reshape(len(sub_sample), 3, len(self._symbols_const))
        return torch.from_numpy(sub_sample.astype('float32'))#.permute(2, 1, 0)
    
    def render(self, mode: str = 'human'):
        raise NotImplementedError
    
    def _get_info(self) -> dict:
        return {
            'portfolio': self._portfolio
        }
