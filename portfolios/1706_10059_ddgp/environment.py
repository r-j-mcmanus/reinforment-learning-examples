import hashlib
import random
import os
from pathlib import Path
from collections import deque
from contextlib import contextmanager
from operator import mul
from functools import reduce

from pathlib import Path
import datetime
from datetime import datetime as dt
from datetime import timedelta

import pandas as pd
import numpy as np
import torch

import mplfinance as mpf
import matplotlib.pyplot as plt


from gymnasium import Env
from gymnasium import spaces

import yfinance as yf

from constants import DEVICE, CONSTANTS

# look into Databento

import matplotlib
matplotlib.use("Agg")


class Sphere(spaces.Box):
    """Normalised action vectors of length one"""
    def sample(self):
        # Dirichlet produces positive vectors summing to 1
        alpha = np.ones(self.shape[0], dtype=np.float32)
        return np.random.dirichlet(alpha).astype(self.dtype) * self.high


class EnvRenderer:
    def __init__(self, env, history_len: int | None):
        # in cpp terms, treat env as a friend
        self._env = env
        self._plot_dir = 'portfolios/1706_10059_ddgp/plots/'
        plot_path = Path(self._plot_dir)
        if not plot_path.exists():
            os.mkdir(plot_path)
        self._history_len = history_len
        if history_len is None:
            self._past_portfolios = []
            self._rewards = []
            self._portfolio_change = []
        elif isinstance(history_len, int):
            self._past_portfolios = deque([], maxlen=history_len)
            self._rewards = deque([], maxlen=history_len)
            self._portfolio_change = deque([], maxlen=history_len)

    def _add(self, portfolio, reward, portfolio_change):
        self._past_portfolios.append(portfolio)
        self._rewards.append(reward)
        self._portfolio_change.append(portfolio_change)

    def update_plots(self, portfolio, reward, portfolio_change):
        self._add(portfolio, reward, portfolio_change)
        lower_bound = max(self._env._obs_index-self._history_len, self._env._initial_obs_index)
        df = self._env._raw_data.set_index('Datetime').iloc[lower_bound: self._env._obs_index]
        self._make_symbol_plots(df)
        self._make_portfolio_factor_plot(df)
        self._make_reward_plot(df)

    def _make_symbol_plots(self, df: pd.DataFrame):
        df_port = pd.DataFrame([i for i in self._past_portfolios])
        df_port.index = df.index
        df_port.columns = ['cash', *self._env._symbols_const]

        for symbol in self._env._symbols_const:
            ap = mpf.make_addplot(
                df_port[symbol],
                type="line",
                color="blue",
                width=1.5
            )

            mpf.plot(
                df.xs(symbol, axis=1, level=1), 
                type="candle", 
                style="charles", 
                volume=False,
                addplot=ap,
                savefig=f"{self._plot_dir}/{symbol}.png"
                )
            plt.close()

    def _make_portfolio_factor_plot(self,  df: pd.DataFrame):
        portfolio_factors = []
        portfolio_factor = 1
        for i in range(len(self._portfolio_change)):
            portfolio_factor *= self._portfolio_change[i]
            portfolio_factors.append(portfolio_factor)
        ds_portfolio_factor = pd.Series(portfolio_factors)
        ds_portfolio_factor.index = df.index
        ax = ds_portfolio_factor.plot(title='Portfolio ratio')
        fig = ax.get_figure()
        fig.savefig(f"{self._plot_dir}/portfolio_factor.png", dpi=150, bbox_inches="tight")
        plt.close()

    def _make_reward_plot(self, df: pd.DataFrame):
        df_reward = pd.Series(self._rewards)
        df_reward.index = df.index
        ax = df_reward.plot(title="Reward Over Time")
        fig = ax.get_figure()
        fig.savefig(f"{self._plot_dir}/reward_plot.png", dpi=150, bbox_inches="tight")
        plt.close()


class AssetTrainingEnvironment(Env):
    def __init__(self,
                 symbols: list[str],
                 history_len: int,
                 start_date: dt = dt(year=2025, month=12, day=20) - timedelta(days=50),
                 end_date: dt = dt(year=2025, month=12, day=20),
                 interval: str = '30m',
                 commission_rate: float = 0.0025 # 0.25%
                ):
        
        super().__init__()
        # consts
        self._sample_history_len = history_len
        self._symbols_const = symbols
        self._commission_rate_const = commission_rate
        # init values
        self.old_portfolio: torch.Tensor = torch.zeros(1).float().to(DEVICE)
        self._set_initial_portfolio()
        
        #for rendering
        self._renderer = EnvRenderer(self, history_len=20)
        self._render = False

        # for sampling
        self._raw_data: pd.DataFrame = self.__init_get_raw_data(symbols, start_date, end_date, interval)
        self._raw_data: pd.DataFrame = self._feature_eng()
        self._symbol_data_on_device: torch.Tensor = self._push_data_to_torch() # in the shape (symbol, history, feature)
        self._data_time: torch.Tensor = self._push_times_to_torch()
        self._subsamples = torch.Tensor([])
        self._str_to_hash: str = ''

        # for calculating reward
        self.old_portfolio = np.zeros(shape=len(self._symbols_const)+1, dtype='float32')
        self.old_portfolio[0] = 1.0

        # for tracking obs location
        self._obs_index: int = 0
        self._initial_obs_index: int = 0
        self._step_number: int = 0

        self.action_space = Sphere(
            low=0.0,
            high=1.0,
            shape=(len(symbols) + 1,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(symbols), history_len, 3), # features: close, high, low, hour
            dtype=np.float32
        )

    def _print_info_on_raw_data(self, batch_size: int):
        print(f'Batch Size {batch_size}, recommended size = root(len(data)) {np.sqrt(len(self._raw_data))}')

    def _feature_eng(self) -> pd.DataFrame:
        # TODO daylight saving messes this up by an hour
        # can pass opening and closing time to not use min and max
        df = self._raw_data
        df = df.reset_index()
        df['Hour'] = df['Datetime'].dt.hour + df['Datetime'].dt.minute/60 
        df["Hour"] = (df["Hour"] - df["Hour"].min()) / (df["Hour"].max() - df["Hour"].min())
        return df

    def __init_get_raw_data(self,
                            symbols: list[str],
                            start_date: dt, 
                            end_date: dt,
                            interval: str) -> pd.DataFrame:
        """
        Download the historic stock data
        """
        self._str_to_hash = str((*symbols, start_date, end_date, interval))
        input_hash = hashlib.sha224(self._str_to_hash.encode()).hexdigest()
        raw_data_file_path = Path(f'portfolios/1706_10059/data/ asset_env_raw_data_{input_hash}.feather')
        if raw_data_file_path.exists():
            print(f'Using saved data {raw_data_file_path}')
            raw_data = pd.read_feather(raw_data_file_path)
        else:
            raw_data = yf.download(symbols, start=start_date, end=end_date, interval=interval)
            assert isinstance(raw_data, pd.DataFrame), "raw data downloaded is not a dataframe"
            if isinstance(raw_data.columns, pd.MultiIndex):
                raw_data.columns.names = ['Field', 'Ticker']
            else:
                raise ValueError("Expected MultiIndex columns from yfinance")

            raw_data.to_feather(raw_data_file_path)
        if raw_data.index[0].tz == None:
            raise Exception('No time zone!')
        if raw_data.index[0].tz != datetime.timezone.utc:
            raw_data.index = raw_data.index.tz_convert("UTC")
        return raw_data
    
    def _push_times_to_torch(self) -> torch.Tensor:
        return torch.from_numpy(self._raw_data['Hour'].values).float().to(DEVICE)

    def _push_data_to_torch(self) -> torch.Tensor:
        """Moves the historical data from the dataframe onto the device in a tensor of shape (symbol, history, feature)"""
        df = self._raw_data.drop(['Open', 'Volume'], axis=1)

        symbol_data_list: list[np.ndarray] = []
        for sym in self._symbols_const:
            close_high_low: np.ndarray = df[:][sym].values
            symbol_data_list.append(close_high_low)
        symbol_data = np.stack(symbol_data_list)
        # in the shape (symbol, history, feature)
        f = torch.from_numpy(symbol_data).float().to(DEVICE)
        return f
        # normalised time since open as feature
        # t = torch.from_numpy(df['Hour'].values).float().to(DEVICE)[None, :, None].expand(8, 449, 1)
        # return torch.cat([f, t], dim=-1).float().to(DEVICE)

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool, bool, dict]:
        """
        1. Move to the next datapoint in the stock data 
        2. Calculate the rewards which is the return on the portfolio (action) for this time step and 
           minus the cost of changing the portfolio from the old
        """
        assert torch.isclose(action.sum(), torch.tensor(1.0), atol=1e-4).all(), "portfolio is not normalised"
        
        # get portfolio
        portfolio = action

        # move time forward
        self._obs_index += 1
        self._step_number += 1
        obs = self._get_obs()

        # how much did the new portfolio earn
        reward , portfolio_change= self._get_reward(portfolio)

        # store for rendering
        if self._render:
            self._past_portfolios.append(portfolio.detach().float().numpy())
            self._rewards.append(reward.detach().float().numpy())
        
        # store for next step
        self.old_portfolio = portfolio.detach()
        
        # output data
        info = self._get_info()
        terminated = self._obs_index >= len(self._raw_data) - 1
        truncated = False

        # sanity
        self._assert_obs_shape(obs)

        if self._render:
            self._renderer.update_plots(portfolio.detach().numpy(), float(reward), portfolio_change)

        return obs, reward, terminated, truncated, info
    
    def _get_reward(self, portfolio: torch.Tensor) -> tuple[torch.Tensor, float]:
        """The rate of return for the time period"""
        mu = self._get_transaction_remainder_factor(portfolio)
        y = self._get_price_relative_vector()
        portfolio_change = mu * torch.dot(y, portfolio)
        reward = torch.log(portfolio_change)
        return reward, float(portfolio_change)
    
    def _get_transaction_remainder_factor(self, portfolio: torch.Tensor):
        """The fractional cost to transition from one portfolio to another portfolio"""
        # TODO this comes from an iterative eq, this is just the initial value
        #   implement the rest
        lost_factor = self._commission_rate_const * torch.sum(torch.abs(portfolio - self.old_portfolio), dim=-1)
        assert (0 <= lost_factor).all()
        assert (lost_factor <= 1).all()
        return 1 - lost_factor

    def _get_price_relative_vector(self) -> torch.Tensor:
        """Data provided is close, high and low, normalise this data wrt the final close"""
        assert isinstance(self._obs_index, int) 
        assert self._obs_index > self._sample_history_len
        assert self._obs_index < len(self._raw_data)


        vt: pd.Series[random.Any] = self._raw_data.iloc[self._obs_index]['Close']
        vtm1 = self._raw_data.iloc[self._obs_index - 1]['Close']

        yt = vt.values / vtm1.values

        return torch.concat([
            torch.ones(1), 
            torch.from_numpy(yt.astype('float32'))
            ]).float().to(DEVICE)
    
    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[torch.Tensor, dict]:
        """Reset portfolio and select new starting observation time
        
        Returns
        -------
        obs: Tensor - shape (Batch, History, Features, Symbols)
        """
        self._step_number = 0

        if isinstance(seed, int):
            super().reset(seed=seed)
        
        self._set_initial_portfolio() # rest to cash only
        self._past_portfolios = [self.old_portfolio]
        self._rewards = []
        self._portfolio_change = []

        self._obs_index = options.get('obs_index') if isinstance(options, dict) else None
        if self._obs_index is None:
            self._obs_index = self._sample_history_len # start far enough in that we have the history
        else:
            if not isinstance(self._obs_index, int) or self._obs_index < self._sample_history_len or self._obs_index >= len(self._raw_data):
                assert ValueError, f'invalid observation_index in options {self._obs_index}'
        
        self._initial_obs_index = self._obs_index # to track how many steps have been taken
        obs = self._get_obs()
        info = self._get_info() # get the info dict
        self._assert_obs_shape(obs)
        return obs, info
    
    def get_train_length(self) -> int:
        return len(self._raw_data) - self._initial_obs_index

    def _get_obs(self) -> torch.Tensor:
        obs = self._get_subsample() # get the data ending at _obs_index
        obs = self._obs_to_normalised_obs(obs)
        return obs

    def _obs_to_normalised_obs(self, obs: torch.Tensor) -> torch.Tensor:
        last_close_price = obs[:,:,-1,0]
        norm_obs = obs / last_close_price[:, :, None, None]
        return norm_obs

    def _set_initial_portfolio(self):
        """Cash only"""
        self.old_portfolio = torch.zeros(len(self._symbols_const)+1).float().to(DEVICE)
        self.old_portfolio[0] = 1.0

    def _get_subsample(self) -> torch.Tensor:
        """Get the n rows before the observation date
        
        subsamples shape = (Batch, Symbols, History, Features)
        """
        assert 0 < self._obs_index < len(self._raw_data)

        # sub-sample length
        L = self._sample_history_len        
        # we want it in reverse order in steps of -1
        offsets =  torch.arange(L - 1, -1, -1, device=DEVICE)  # (L,)
        # the indexes for each sub-sample in the batch
        hist_idx = self._obs_index - offsets[None, :]  # (B, L)
        # Reshape to get the shape we want
        out = self._symbol_data_on_device[:, hist_idx, :].permute(1, 0, 2, 3)

        return out
    
    @contextmanager
    def render(self, mode: str = 'human'):
        self._render = True

    def _get_info(self) -> dict:
        return {}

    def _assert_obs_shape(self, obs):
        assert obs.shape[1:] == self.observation_space.shape, 'Obs shape doesn\'t match that specified in self.observation_space'

    def get_batch_sample(self, obs_idx: list[int]) -> torch.Tensor:
        """
        Args:
            obs_idx (list[int]): List of indices for the end of the training samples for each env observation
        """
        obs_index: torch.Tensor = torch.tensor(obs_idx).int().to(DEVICE)
        # how much history is provided as part of the observation
        L = self._sample_history_len
        # we want it in reverse order in steps of -1
        offsets = torch.arange(L - 1, -1, -1, device=DEVICE)  # (L,)
        # the indexes for each sub-sample in the batch
        hist_idx = obs_index[:, None] - offsets[None, :]  # (B, L)
        # Reshape to get the shape we want
        obs_batch = self._symbol_data_on_device[:, hist_idx, :].permute(1, 0, 2, 3)
        
        last_close_price = obs_batch[:,:,-1,0]
        norm_obs_batch = obs_batch / last_close_price[:, :, None, None]

        return norm_obs_batch