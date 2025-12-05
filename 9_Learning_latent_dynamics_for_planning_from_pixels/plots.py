import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime as dt

from pathlib import Path

_DATE = dt.now().strftime("%Y%m%d%H%M")

_PATH = '9_Learning_latent_dynamics_for_planning_from_pixels'

def plot_rssm_data(df: pd.DataFrame, episode: int):
    print(f'making plots {episode}')
    with plt.ioff():
        _plot_df(df, episode, 'rssm_loss')
    print(f'finished plots {episode}')

def plot_ll_data(df: pd.DataFrame, episode: int):
    print(f'making plots {episode}')
    with plt.ioff():
        _plot_df(df, episode, 'actor_critic_loss')
    print(f'finished plots {episode}')

def plot_env_data(df: pd.DataFrame, episode: int):
    print(f'making plots {episode}')
    with plt.ioff():
        _plot_df(df, episode, 'env')
    print(f'finished plots {episode}')

def _plot_df(df: pd.DataFrame, episode: int, folder_name: str):

    fig_path = Path(f'{_PATH}/fig/{_DATE}/{folder_name}/{episode}')
    fig_path.mkdir(parents=True, exist_ok=True)
    
    for col in df.columns:
        if col == 'episode' or col == 'epoch':
            continue

        plt.figure()
        df[col].plot(title=col)
        plt.xlabel("Index")
        plt.ylabel(col)
        plt.tight_layout()
        
        save_path = fig_path / f"{col}.png"
        plt.savefig(save_path)
        plt.close()
