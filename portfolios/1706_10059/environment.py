import hashlib
import random
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

class AssetTrainingEnvironment(Env):
    def __init__(self,
                 symbols: list[str],
                 start_date: dt = dt(year=2025, month=12, day=20) - timedelta(days=50),
                 end_date: dt = dt(year=2025, month=12, day=20),
                 interval: str = '30m',
                 commission_rate: float = 0.0025 # 0.25%
                ):
        
        super().__init__()
        # consts
        self._input_period_len_const = CONSTANTS.INPUT_PERIOD_LEN
        self._symbols_const = symbols
        self._max_steps_const = CONSTANTS.MAX_STEPS
        self._commission_rate_const = commission_rate
        # init values
        self.old_portfolio: torch.Tensor = torch.zeros(1).float().to(DEVICE)
        self._set_initial_portfolio()
        
        #for rendering
        self._render = False
        self._past_portfolios = []
        self._rewards = []
        self._portfolio_change = []

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
            shape=(len(symbols), CONSTANTS.INPUT_PERIOD_LEN, 3), # features: close, high, low, hour
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
        t = torch.from_numpy(df['Hour'].values).float().to(DEVICE)[None, :, None].expand(8, 449, 1)
        return torch.cat([f, t], dim=-1).float().to(DEVICE)

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool, bool, dict]:
        """
        1. Move to the next datapoint in the stock data 
        2. Calculate the rewards which is the return on the portfolio for this time step and 
           minus the cost of changing the portfolio from the old
        """
        assert torch.isclose(action.sum(dim=-1), torch.tensor(1.0), atol=1e-4).all(), "portfolio is not normalised"
        
        # get portfolio
        portfolio = action

        # move time forward
        self._obs_index = [i+1 for i in self._obs_index]
        self._step_number += 1
        obs = self._get_obs()

        # how much did the new portfolio earn
        reward = self._get_reward(portfolio)

        # store for rendering
        if self._render:
            self._past_portfolios.append(portfolio.detach().float().numpy()[0])
            self._rewards.append(reward.detach().float().numpy()[0])
        
        # store for next step
        self.old_portfolio = portfolio.detach()
        
        # output data
        info = self._get_info()
        terminated = self._step_number >= self._max_steps_const -1
        truncated = False
        # sanity
        self._assert_obs_shape(obs)
        return obs, reward, terminated, truncated, info
    
    def _get_reward(self, portfolio: torch.Tensor) -> torch.Tensor:
        """The rate of return for the time period"""
        mu = self._get_transaction_remainder_factor(portfolio)
        y = self._get_price_relative_vector()
        portfolio_change = mu * (y * portfolio).sum(dim=1)
        if self._render:
            self._portfolio_change.append(float(portfolio_change.detach().numpy()[0]))
        reward = torch.log(portfolio_change)
        return reward #/ self._max_steps_const
    
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

        batch_size =  options.get('batch_size', 1)
        assert isinstance(batch_size, int)
        # self._print_info_on_raw_data(batch_size)

        if isinstance(seed, int):
            super().reset(seed=seed)
        
        self._set_initial_portfolio() # rest to cash only
        self._past_portfolios = [self.old_portfolio]
        self._rewards = []
        self._portfolio_change = []

        # get indices that have at least self._input_period_len_const entries behind them
        self._obs_index = options.get('obs_index')
        if self._obs_index is None:
            self._obs_index = [
                i + self._input_period_len_const for i in 
                random.sample( # pick without replacement
                    range(
                        len(self._raw_data) - self._input_period_len_const - self._max_steps_const
                    ) # cannot pick so early that we dont have a history, nor late we dont have an episode 
                    , batch_size)
                ]
        else:
            if isinstance(self._obs_index, int):
                self._obs_index = [self._obs_index]
            else:
                assert isinstance(self._obs_index, list)
                assert len(self._obs_index)>0
                for i in self._obs_index:
                    assert isinstance(i, int)
        
        self._initial_obs_index = self._obs_index # to track how many steps have been taken
        obs = self._get_obs()
        info = self._get_info() # get the info dict
        self._assert_obs_shape(obs)
        return obs, info
    
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
        assert len(self._obs_index) > 0

        # symbols, history len, features
        S, H, F = self._symbol_data_on_device.shape
        # batch size
        B = len(self._obs_index)
        # sub-sample length
        P = self._input_period_len_const
        # future steps
        T = self._max_steps_const
        # how long is the subsample
        L = P
        
        obs_index: torch.Tensor = torch.tensor(self._obs_index).int().to(DEVICE)
        # we want it in reverse order in steps of -1
        offsets = torch.arange(L - 1, -1, -1, device=DEVICE)  # (L,)
        # the indexes for each sub-sample in the batch
        hist_idx = obs_index[:, None] - offsets[None, :]  # (B, L)
        # Reshape to get the shape we want
        out = self._symbol_data_on_device[:, hist_idx, :].permute(1, 0, 2, 3)

        return out
    
    @contextmanager
    def render(self, mode: str = 'human'):
        self._render = True
        yield self
        self._render = False
        self._make_plots()

    def _make_plots(self):
        df: pd.DataFrame = self._raw_data.set_index('Datetime').iloc[self._initial_obs_index[0]:self._initial_obs_index[0]+self._step_number+1]
        df_port = pd.DataFrame([i for i in self._past_portfolios])
        df_port.index = df.index
        df_port.columns = ['cash', *self._symbols_const]

        for symbol in self._symbols_const:
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
                savefig=f"portfolios/1706_10059/plots/{symbol}.png"
                )
            plt.close()
            
        df_reward = pd.Series(self._rewards)
        df_reward.index = df.index[1:]
        ax = df_reward.plot(title="Reward Over Time")
        fig = ax.get_figure()
        fig.savefig("portfolios/1706_10059/plots/reward_plot.png", dpi=150, bbox_inches="tight")
        plt.close()

        ds_portfolio_factor = pd.Series([reduce(mul, self._portfolio_change[0:i], 1) for i in range(len(self._portfolio_change)+1)])
        ds_portfolio_factor.index = df.index
        ax = ds_portfolio_factor.plot(title='Portfolio ratio')
        fig = ax.get_figure()
        fig.savefig("portfolios/1706_10059/plots/portfolio_factor.png", dpi=150, bbox_inches="tight")
        plt.close()

    def _get_info(self) -> dict:
        return {}

    def _assert_obs_shape(self, obs):
        assert obs.shape[1:] == self.observation_space.shape, 'Obs shape doesn\'t match that specified in self.observation_space'
