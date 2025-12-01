import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime as dt

from pathlib import Path

_DATE = dt.now().strftime("%Y%m%d%H%M")

_PATH = '9_Learning_latent_dynamics_for_planning_from_pixels'

def plot_rssm_data(df: pd.DataFrame, episode: int):
    print(f'making plots {episode}')
    with plt.ioff():
        _plot_rssm_losses(df, episode)
    print(f'finished plots {episode}')

def _plot_rssm_losses(df: pd.DataFrame, episode: int):

    fig_path = Path(f'{_PATH}/fig/{_DATE}/rssm_loss/{episode}')
    fig_path.mkdir(parents=True, exist_ok=True)
    
    for col in df.columns:
        if col == 'episode' or col == 'epoch':
            continue

        delta = df[col].max() - df[col].min()

        # Format the number for filenames (avoid spaces, long floats, etc.)
        delta = f"{delta:.4g}"   # 4 significant digits, compact
        delta = str(delta).replace("/", "_")  # basic safety

        plt.figure()
        df[col].plot(title=col)
        plt.xlabel("Index")
        plt.ylabel(col)
        plt.tight_layout()
        
        save_path = fig_path / f"{col}_{delta}.png"
        plt.savefig(save_path)
        plt.close()
