import numpy as np
import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch import distributions

from flows import RealNVP, get_mask


class StructuredLatentFlowModel:
    """
    Combines a pretrained STL-based autoencoder with a RealNVP flow model over its latent space.

    Components:
    - Encoder: fₑ(x) → z ∈ ℝ^d, with structured trend-seasonal decomposition.
    - Flow model: g(z) = u, where u ~ N(0, Σ), learned to match latent distribution to prior.
    - Prior: Multivariate normal N(0, Σ) with low-dimensional signal + isotropic noise tail.
             Given by the matrix cov_matrix
             
        hidden_d: dimensionality of embeddings (output size of ae_model)
        T: length of the flow
        aux: dimensionality of the flow

    After training, flow log-density log p(z) = log p(u) + log|det ∂g/∂z| provides likelihood estimate in latent space.
    """
    def __init__(self, ae_model, device, cov_matrix, T=55, nets=None, nett=None):
        self.device = device
        self.hidden_d = cov_matrix.shape[-1]
        self.T = T

        # Load pretrained autoencoder
        self.model = ae_model.double().to(device)
        self.model.eval()

        self.cov = cov_matrix

        # Define prior and flow model
        self.prior = torch.distributions.MultivariateNormal(
            torch.zeros(self.hidden_d).double().to(device), self.cov)
        
        self.flow = RealNVP(
            nets=nets,
            nett=nett,
            mask=torch.Tensor(get_mask(self.hidden_d, T)),
            prior=self.prior
        ).to(device).double()

    def train_flow(self, loader, epochs=100, lr=7e-5, gamma=0.95, log_interval=100):
        """
        Trains the flow model to match the latent posterior of the autoencoder.

        Args:
            loader (DataLoader): Batches of input data (X, target).
            epochs (int): Number of training epochs.
            lr (float): Learning rate for Adam optimizer.
            gamma (float): LR scheduler decay factor.
            log_interval (int): Batch logging frequency.
        """
        self.flow.train()
        optimizer = torch.optim.Adam(self.flow.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

        for epoch in range(1, epochs + 1):
            for batch_idx, (data, _) in enumerate(loader):
                z = self.model.encode(data.double().to(self.device))

                optimizer.zero_grad()
                loss = -self.flow.log_prob(z).sum()
                loss.backward()
                optimizer.step()

                if batch_idx % log_interval == 0:
                    print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.3f}")
            scheduler.step()

    def encode(self, x):
        return self.model.encode(x.to(self.device).double())

    def decode(self, z):
        return self.model.decode(z.to(self.device).double())

    def log_likelihood(self, x):
        """
        Estimate log-likelihood in latent space using flow
        """
        z = self.encode(x)
        return self.flow.log_prob(z)
    
    
class ClassCenters:
    """
    Defines fixed latent vectors (centroids) for each class label.

    Args:
        n_classes (int): Number of classes.
        dim (int): Embedding dimension.
        typ (str): Type of encoding:
            - 'OHE': one-hot repeated across dim
            - 'sphere': normalized random vectors (Gaussian)
            - 'un': normalized uniform vectors in [-1, 1]

    Method:
        get_encodes(target): Returns encodings for a batch of class indices.
    """
    def __init__(self, n_classes, dim, typ):
        # Initialize class centers based on encoding type
        if typ == 'OHE':
            self.cent = torch.eye(n_classes).repeat(1, dim)  # One-hot repeated across dim
        elif typ == 'sphere':
            cent = torch.randn(n_classes, dim)               # Random normals
            self.cent = cent / cent.norm(dim=1, keepdim=True)  # Normalize to unit sphere
        elif typ == 'un':
            cent = torch.distributions.Uniform(-1, 1).sample((n_classes, dim))
            self.cent = cent / cent.norm(dim=1, keepdim=True)
        else:
            raise ValueError(f"Unknown type: {typ}")

    def get_encodes(self, target):
        # Return encoding for each class label
        return self.cent[target.long()]
