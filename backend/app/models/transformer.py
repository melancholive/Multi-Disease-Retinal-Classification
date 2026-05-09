import torch
import torch.nn as nn


class TransformerAttentionFusion(nn.Module):
    def __init__(
        self,
        dim_shuffle=1024,
        dim_eff=1280,
        dim_res=2048,
        embed_dim=512,
        num_heads=8,
        num_transformer_layers=1,
        transformer_dropout=0.1,  # define input drop out at 0.1
    ):
        super().__init__()

        # Project each CNN feature vector to common embedding size
        self.proj_shuffle = nn.Linear(dim_shuffle, embed_dim)
        self.proj_eff = nn.Linear(dim_eff, embed_dim)
        self.proj_res = nn.Linear(dim_res, embed_dim)

        # Normalize projected embeddings
        self.norm_shuffle = nn.LayerNorm(embed_dim)
        self.norm_eff = nn.LayerNorm(embed_dim)
        self.norm_res = nn.LayerNorm(embed_dim)

        # Learnable token identity embeddings
        self.token_type_embed = nn.Parameter(torch.randn(1, 3, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=transformer_dropout,
            activation="relu",  # relu activation function
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )

        # Attention scoring MLP: outputs 1 score per token
        self.attn_mlp = nn.Sequential(
            nn.Linear(embed_dim, 128),  # input layer
            nn.ReLU(),  # relu activation function
            nn.Linear(128, 1),  # output layer
        )

    def forward(self, f_shuffle, f_eff, f_res):
        """
        f_shuffle: [B, 1024]
        f_eff:     [B, 1280]
        f_res:     [B, 2048]
        """

        z_shuffle = self.norm_shuffle(self.proj_shuffle(f_shuffle))  # [B, 512]
        z_eff = self.norm_eff(self.proj_eff(f_eff))  # [B, 512]
        z_res = self.norm_res(self.proj_res(f_res))  # [B, 512]

        tokens = torch.stack([z_shuffle, z_eff, z_res], dim=1)  # [B, 3, 512]
        # each CNN as 1 token

        # 3. Add token identity embeddings
        tokens = tokens + self.token_type_embed  # [B, 3, 512]

        # 4. Transformer encoder
        tokens = self.transformer(tokens)  # [B, 3, 512]

        # 5. Attention scores
        scores = self.attn_mlp(tokens).squeeze(-1)  # [B, 3]

        # 6. Softmax weights
        weights = torch.softmax(scores, dim=1)  # [B, 3]

        # 7. Weighted sum fusion
        fused = torch.sum(tokens * weights.unsqueeze(-1), dim=1)  # [B, 512]
        # attention scores derived from transformer
        # use attention scores to weigh features from CNNs

        return fused, weights, tokens, scores


class PredictionBlock(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_classes=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)
