import torch
import torch.nn as nn

class TabularSeverityNet(nn.Module):
    def __init__(self, num_numerics: int, cat_cardinalities: list, embedding_dims: list):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=card, embedding_dim=dim)
            for card, dim in zip(cat_cardinalities, embedding_dims)
        ])
        total_embed_dim = sum(embedding_dims)
        
        input_dim = num_numerics + total_embed_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x_num, x_cat):
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        if embedded:
            x_cat_concat = torch.cat(embedded, dim=1)
            x = torch.cat([x_num, x_cat_concat], dim=1)
        else:
            x = x_num
        
        out = self.mlp(x)
        # Log-link output head for positive severity
        return torch.exp(out)

class GammaNLLLoss(nn.Module):
    """Negative Log-Likelihood loss for Gamma distribution with log-link mean."""
    def __init__(self, shape: float = 2.0):
        super().__init__()
        self.shape = shape

    def forward(self, y_pred, y_true):
        # y_pred is mean mu, y_true is target y
        # Gamma NLL: y / mu + log(mu) [ignoring constant terms]
        eps = 1e-6
        mu = torch.clamp(y_pred, min=eps)
        y = torch.clamp(y_true, min=0.0)
        loss = y / mu + torch.log(mu)
        return torch.mean(loss)
