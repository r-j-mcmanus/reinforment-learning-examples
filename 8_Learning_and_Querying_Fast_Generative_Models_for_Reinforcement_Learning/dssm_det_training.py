import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dssm_det import DSSM_DET


def train_dssm(model: DSSM_DET, dataloader, num_epochs=10, lr=1e-3, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for obs, action, target_state in dataloader:
            obs = obs.to(device)  # (seq_len, batch_size, obs_size)
            action = action.to(device)  # (seq_len, batch_size, action_size)
            target_state = target_state.to(device)  # (seq_len, batch_size, hidden_size)

            optimizer.zero_grad()
            pred_state = model(obs, action)  # (seq_len, batch_size, hidden_size)
            loss = criterion(pred_state, target_state)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")


if __name__ == '__main__':
    # Parameters
    seq_len = 10 # length of each sequence we train the rnn on 
    batch_size = 16 # how many sequences per batch
    num_sequences = 100 # total number of sequences
    obs_size = 8 # the dimention of the latent space
    action_size = 4 # how many actions the agent can take
    hidden_size = obs_size # as we use the hidden state as the latent state

    # Generate synthetic data
    obs_data = torch.randn(seq_len, num_sequences, obs_size)
    action_data = torch.randn(seq_len, num_sequences, action_size)
    target_state_data = torch.randn(seq_len, num_sequences, hidden_size)

    # Create TensorDataset and DataLoader
    dataset = TensorDataset(obs_data, action_data, target_state_data)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Initialize model
    model = DSSM_DET(state_size=hidden_size, obs_size=obs_size, action_size=action_size, hidden_size=hidden_size)

    train_dssm(model, dataloader, num_epochs=20, lr=1e-3)