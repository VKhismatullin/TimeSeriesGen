import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm

def L2_series(trends, x):
    # Project input x onto a set of basis trends via inner product
    return torch.sum(trends * x, dim=1)

class SeasonalTrendLatentEncoder(nn.Module):
    """
    Encodes multivariate time series x ∈ ℝ^{L×M} into latent vectors z ∈ ℝ^d by projecting onto predefined basis functions.

    The model uses:
    - k pairs of sine and cosine harmonics (seasonality) defined over [0, L)
    - poly-degree polynomial basis (trend) over normalized time
    - A parallel fully connected pathway for direct regression over raw input

    Output is z = f_proj(x) + f_raw(x), where 
            f_proj(x) captures structured frequency and polynomial content, 
            f_raw(x) captures unstructured residual variation.      
            
    Key Parameters:
    - L (int): sequence length
    - M (int): number of features per time step
    - d (int): latent dimension
    - k (int): number of harmonics (for both sine and cosine)
    - poly (int): number of polynomial basis functions
    - aux_dim (int): hidden dimension for MLPs
    - denom (int): frequency scaling factor (typically 1)


    Usage:
        encoder = SeasonalTrendLatentEncoder(L=30, M=26, d=64, k=10, poly=12)
        z = encoder(x)  # z ∈ ℝ^{N × d}
    """
    def __init__(self, aux_dim, int_dim, d, k, denom, L, m, poly, device='cpu'):
        super().__init__()
        self.aux_dim = aux_dim
        self.int_dim = int_dim
        self.d = d
        self.k = k
        self.denom = denom
        self.L = L
        self.m = m
        self.poly = poly

        self.reg1 = nn.Sequential(
            nn.Linear(self.m * self.L, self.aux_dim), nn.ReLU(),
            nn.Linear(self.aux_dim, self.aux_dim), nn.ReLU(),
            nn.Linear(self.aux_dim, self.aux_dim), nn.ReLU(),
            nn.Linear(self.aux_dim, self.d)
        )

        self.series_coder = nn.Sequential(
            nn.Linear(self.m * self.k * 2 + self.m * self.poly, self.aux_dim), nn.ReLU(),
            nn.Linear(self.aux_dim, self.aux_dim), nn.ReLU(),
            nn.Linear(self.aux_dim, self.aux_dim), nn.ReLU(),
            nn.Linear(self.aux_dim, self.d)
        )

        self.trends1 = torch.cos(torch.arange(self.L).reshape(1, -1, 1, 1) / self.denom / self.L * 2 * np.pi *
                                 torch.arange(0, self.k).reshape(1, 1, -1, 1)).repeat(1, 1, 1, self.m).to(device)
        self.trends2 = torch.sin(torch.arange(self.L).reshape(1, -1, 1, 1) / self.denom / self.L * 2 * np.pi *
                                 torch.arange(1, self.k + 1).reshape(1, 1, -1, 1)).repeat(1, 1, 1, self.m).to(device)
        self.trends3 = (torch.arange(self.L).reshape(1, -1, 1, 1) / self.L) ** torch.arange(1, self.poly + 1).reshape(1, 1, -1, 1)
        self.trends3 = self.trends3.repeat(1, 1, 1, self.m).to(device)

    def forward(self, x):
        # Input: (N x L x M)
        h1 = x.reshape(-1, self.L, 1, self.m)
        h1_proj = torch.cat((
            L2_series(self.trends1, h1).flatten(1),
            L2_series(self.trends2, h1).flatten(1),
            L2_series(self.trends3, h1).flatten(1)
        ), dim=1)
        seasonal_feat = self.series_coder(h1_proj)
        reg_feat = self.reg1(x.flatten(1))
        return seasonal_feat + reg_feat

    
class STLStructuredAutoencoder(nn.Module):
    """
    STLStructuredAutoencoder is a neural architecture for decomposing multivariate time series 
        into trend and seasonal components in a structured and interpretable way. 

    The model decomposes x as:
        x ≈ trend(x) + season(x)
        
    The input is assumed to be a sequence x ∈ ℝ^{L×M}, where:
        - L is the number of time steps (sequence length)
        - M is the number of features (e.g., sensor channels or joint positions)

    Key components:

    1. **Encoder**:
        - Projects the input sequence x into a latent representation z ∈ ℝ^d.
        - It is based on the STL decomposition

    2. **Trend Decoder**:
        - Reconstructs the trend component from the latent vector z using a polynomial basis.
        - Each time step is represented as a polynomial combination: 
              trend(x) = Σ_{i=0}^{poly-1} β_i · (t / L)^i
          where β_i are learned coefficients, and t is normalized time.
        - The coefficients are predicted from z using a neural decoder.

    3. **Seasonal Encoders and Decoders**:
        - The reconstructed trend is passed through the encoder again to obtain z_trend.
        - z_trend is subtracted from the original latent z to isolate the residual seasonal component.
        - Two parallel decoders map this seasonal latent representation to:
            - Cosine coefficients: α_j in Σ α_j · cos(2πjt/L)
            - Sine coefficients:   γ_k in Σ γ_k · sin(2πkt/L)
        - The resulting cosine and sine reconstructions are summed to form the seasonal signal.

    4. **Reconstruction**:
        - The final output is the sum of the reconstructed trend and seasonal signals.
        - This structure ensures that different latent subspaces are responsible for trend and seasonal variations.


    Parameters:
        - L (int): sequence length
        - M (int): feature dimension per time step
        - d (int): latent dimension
        - poly (int): number of polynomial terms used in the trend
        - n (int) number of frequency terms (typically coupled with L)
        - aux_enc (int): hidden dimension of encoder
        - aux_dec (int): hidden dimension of decoders
        - denom (int): frequency scaling factor (typically 1)
        
        

    Usage:
        model = STLStructuredAutoencoder(L=30, M=26, d=64, poly=12)
        output = model(x)  # where x ∈ ℝ^{batch_size × L × M}
    """
    def __init__(self, encoder_kwargs, L, M, d, poly, n, aux_enc, aux_dec, denom=1, device='cpu'):
        super().__init__()
        self.l = L
        self.m = M 
        self.d = d
        self.poly = poly
        self.n = n
        self.aux_enc = aux_enc
        self.aux_dec = aux_dec
        self.denom = denom
        self.device = device
        
                  # number of cosine and sine terms
        
        
        self.encoder = SeasonalTrendLatentEncoder(**encoder_kwargs)

        self.lindecoder1 = nn.Sequential(
            nn.Linear(self.d, self.aux_dec), nn.ELU(),
            nn.Linear(self.aux_dec, self.aux_dec), nn.ELU(),
            nn.Linear(self.aux_dec, self.m * self.n)
        )

        self.lindecoder2 = nn.Sequential(
            nn.Linear(self.d, self.aux_dec), nn.ELU(),
            nn.Linear(self.aux_dec, self.aux_dec), nn.ELU(),
            nn.Linear(self.aux_dec, self.m * self.n)
        )

        self.trend_coef = nn.Sequential(
            nn.Linear(self.d, self.aux_dec), nn.ELU(),
            nn.Linear(self.aux_dec, self.aux_dec), nn.ELU(),
            nn.Linear(self.aux_dec, self.m * self.poly)
        )

        self.trends1 = torch.cos(torch.arange(self.l).reshape(1, -1, 1, 1) / self.l * 2 * np.pi / self.denom *
                                 torch.arange(0, self.n).reshape(1, 1, -1, 1)).repeat(1, 1, 1, self.m).to(device)
        self.trends2 = torch.sin(torch.arange(self.l).reshape(1, -1, 1, 1) / self.l * 2 * np.pi / self.denom *
                                 torch.arange(1, self.n + 1).reshape(1, 1, -1, 1)).repeat(1, 1, 1, self.m).to(device)
        self.trends3 = ((torch.arange(self.l).reshape(1, -1, 1, 1) / self.l) ** 
                        torch.arange(0, self.poly).reshape(1, 1, -1, 1)).repeat(1, 1, 1, self.m).to(device)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        x = x.reshape(-1, 1, self.d)

        x_trend = torch.sum(self.trends3 * self.trend_coef(x).reshape(-1, 1, self.poly, self.m), dim=2)
        x_no_trend = x - self.encode(x_trend).reshape(-1, 1, self.d)

        seasonal1 = self.lindecoder1(x_no_trend).reshape(-1, 1, self.n, self.m)
        seasonal2 = self.lindecoder2(x_no_trend).reshape(-1, 1, self.n, self.m)

        x_seasonal = torch.sum(self.trends1 * seasonal1, dim=2) + torch.sum(self.trends2 * seasonal2, dim=2)
        return x_seasonal + x_trend

    def forward(self, x):
        return self.decode(self.encode(x))

####
# 3. Basic evaluation and training
####


def evaluate_autoenc(model, device, data_loader, LOSS, desc="Eval"):
    """
    Evaluate autoencoder performance on a dataset.
    Computes average loss across all samples, with a tqdm bar.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for data, _ in tqdm(data_loader, desc=desc, leave=False):
            data = data.float().to(device)
            output = model(data)

            batch_size = data.size(0)
            total_loss += LOSS(
                output.flatten(1), 
                data.flatten(1)
            ).item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


def train_autoenc(
    model,
    device,
    train_loader,
    test_loader,
    n_epoch,
    optimizer,
    scheduler,
    LOSS
):
    """
    Train the autoencoder over multiple epochs.
    Logs training and test losses after each epoch, with tqdm bars.
    """
    train_history = {'loss': []}
    test_history = {'loss': []}

    for epoch in range(1, n_epoch + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{n_epoch}", leave=False)
        for data, _ in pbar:
            data = data.float().to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = LOSS(output.flatten(1), data.flatten(1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * data.size(0)
            pbar.set_postfix_str(f"Batch Loss: {loss.item():.4f}")

        scheduler.step()

        avg_train_loss = epoch_loss / len(train_loader.dataset)
        train_history['loss'].append(avg_train_loss)
        print(f"Epoch {epoch} ▶ Training Loss: {avg_train_loss:.4f}")

        if test_loader is not None:
            avg_test_loss = evaluate_autoenc(
                model, device, test_loader, LOSS, desc=f"Test {epoch}"
            )
            test_history['loss'].append(avg_test_loss)
            print(f"           Test Loss:     {avg_test_loss:.4f}")

    return train_history, test_history

