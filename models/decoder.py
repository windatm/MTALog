import torch
import torch.nn as nn


class MLPDecoder(nn.Module):
    """
    A lightweight decoder module that reconstructs input embeddings
    from latent representations. Useful in unsupervised or semi-supervised
    settings (e.g., autoencoding log event representations).

    Args:
        input_dim (int): Dimension of the latent representation.
        output_dim (int): Dimension of the reconstructed embedding (e.g., 300).
    """
    def __init__(self, input_dim: int, output_dim: int):
        super(MLPDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, output_dim)
        )

    def forward(self, z):
        return self.decoder(z)
