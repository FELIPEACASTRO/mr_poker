"""Style Embedding — Auto-encoder for opponent behavioral stats.

Compresses 12-dimensional opponent statistics into a dense latent
embedding vector. Enables similarity comparison between opponents
and clustering of playing styles.

The auto-encoder is trained online as new opponent data arrives,
learning a compact representation of player tendencies.

Reference: Heinrich & Silver (2016) "Deep Reinforcement Learning
from Self-Play in Imperfect-Information Games"
"""

from __future__ import annotations

import math
import random


class StyleEmbedder:
    """Compresses opponent behavioral stats into dense embeddings.

    Architecture:
        Encoder: input(12) -> hidden(24) -> latent(16) with ReLU
        Decoder: latent(16) -> hidden(24) -> output(12) reconstruction

    Uses pure-Python weights (no external dependencies), matching
    the SimpleNN pattern from deep_cfr.py.
    """

    def __init__(
        self,
        input_dim: int = 12,
        latent_dim: int = 16,
        hidden_dim: int = 24,
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        rng = random.Random(seed)

        # Encoder weights
        scale_e1 = math.sqrt(2.0 / input_dim)
        scale_e2 = math.sqrt(2.0 / hidden_dim)
        self.enc_w1 = [[rng.gauss(0, scale_e1) for _ in range(input_dim)] for _ in range(hidden_dim)]
        self.enc_b1 = [0.0] * hidden_dim
        self.enc_w2 = [[rng.gauss(0, scale_e2) for _ in range(hidden_dim)] for _ in range(latent_dim)]
        self.enc_b2 = [0.0] * latent_dim

        # Decoder weights
        scale_d1 = math.sqrt(2.0 / latent_dim)
        scale_d2 = math.sqrt(2.0 / hidden_dim)
        self.dec_w1 = [[rng.gauss(0, scale_d1) for _ in range(latent_dim)] for _ in range(hidden_dim)]
        self.dec_b1 = [0.0] * hidden_dim
        self.dec_w2 = [[rng.gauss(0, scale_d2) for _ in range(hidden_dim)] for _ in range(input_dim)]
        self.dec_b2 = [0.0] * input_dim

    def encode(self, stats: list[float]) -> list[float]:
        """Encode stats into latent embedding."""
        # Layer 1: input -> hidden (ReLU)
        hidden = []
        for i in range(self.hidden_dim):
            val = self.enc_b1[i] + sum(self.enc_w1[i][j] * stats[j] for j in range(self.input_dim))
            hidden.append(max(0.0, val))

        # Layer 2: hidden -> latent (ReLU)
        latent = []
        for i in range(self.latent_dim):
            val = self.enc_b2[i] + sum(self.enc_w2[i][j] * hidden[j] for j in range(self.hidden_dim))
            latent.append(max(0.0, val))

        return latent

    def decode(self, embedding: list[float]) -> list[float]:
        """Decode embedding back to stats (for training)."""
        # Layer 1: latent -> hidden (ReLU)
        hidden = []
        for i in range(self.hidden_dim):
            val = self.dec_b1[i] + sum(self.dec_w1[i][j] * embedding[j] for j in range(self.latent_dim))
            hidden.append(max(0.0, val))

        # Layer 2: hidden -> output (linear, for reconstruction)
        output = []
        for i in range(self.input_dim):
            val = self.dec_b2[i] + sum(self.dec_w2[i][j] * hidden[j] for j in range(self.hidden_dim))
            output.append(val)

        return output

    def train_step(self, stats: list[float], lr: float = 0.001) -> float:
        """Train auto-encoder on a single sample. Returns reconstruction loss.

        Performs full forward pass through encoder and decoder, then
        backpropagates MSE loss through both networks.
        """
        # === Forward pass (encoder) ===
        enc_hidden_raw = []
        enc_hidden = []
        for i in range(self.hidden_dim):
            val = self.enc_b1[i] + sum(self.enc_w1[i][j] * stats[j] for j in range(self.input_dim))
            enc_hidden_raw.append(val)
            enc_hidden.append(max(0.0, val))

        latent_raw = []
        latent = []
        for i in range(self.latent_dim):
            val = self.enc_b2[i] + sum(self.enc_w2[i][j] * enc_hidden[j] for j in range(self.hidden_dim))
            latent_raw.append(val)
            latent.append(max(0.0, val))

        # === Forward pass (decoder) ===
        dec_hidden_raw = []
        dec_hidden = []
        for i in range(self.hidden_dim):
            val = self.dec_b1[i] + sum(self.dec_w1[i][j] * latent[j] for j in range(self.latent_dim))
            dec_hidden_raw.append(val)
            dec_hidden.append(max(0.0, val))

        output = []
        for i in range(self.input_dim):
            val = self.dec_b2[i] + sum(self.dec_w2[i][j] * dec_hidden[j] for j in range(self.hidden_dim))
            output.append(val)

        # === Loss (MSE) ===
        n = len(stats)
        loss = sum((output[i] - stats[i]) ** 2 for i in range(n)) / n

        # === Backward (decoder output layer) ===
        d_output = [(output[i] - stats[i]) * 2.0 / n for i in range(n)]

        d_dec_hidden = [0.0] * self.hidden_dim
        for i in range(self.input_dim):
            for j in range(self.hidden_dim):
                d_dec_hidden[j] += d_output[i] * self.dec_w2[i][j]
                self.dec_w2[i][j] -= lr * d_output[i] * dec_hidden[j]
            self.dec_b2[i] -= lr * d_output[i]

        # === Backward (decoder hidden layer, ReLU) ===
        d_latent = [0.0] * self.latent_dim
        for i in range(self.hidden_dim):
            if dec_hidden_raw[i] <= 0:
                continue
            grad = d_dec_hidden[i]
            for j in range(self.latent_dim):
                d_latent[j] += grad * self.dec_w1[i][j]
                self.dec_w1[i][j] -= lr * grad * latent[j]
            self.dec_b1[i] -= lr * grad

        # === Backward (encoder latent layer, ReLU) ===
        d_enc_hidden = [0.0] * self.hidden_dim
        for i in range(self.latent_dim):
            if latent_raw[i] <= 0:
                continue
            grad = d_latent[i]
            for j in range(self.hidden_dim):
                d_enc_hidden[j] += grad * self.enc_w2[i][j]
                self.enc_w2[i][j] -= lr * grad * enc_hidden[j]
            self.enc_b2[i] -= lr * grad

        # === Backward (encoder hidden layer, ReLU) ===
        for i in range(self.hidden_dim):
            if enc_hidden_raw[i] <= 0:
                continue
            grad = d_enc_hidden[i]
            for j in range(self.input_dim):
                self.enc_w1[i][j] -= lr * grad * stats[j]
            self.enc_b1[i] -= lr * grad

        return loss

    def similarity(self, emb1: list[float], emb2: list[float]) -> float:
        """Cosine similarity between two embeddings."""
        dot = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = math.sqrt(sum(a * a for a in emb1))
        norm2 = math.sqrt(sum(b * b for b in emb2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    @staticmethod
    def stats_from_player(
        vpip: float,
        pfr: float,
        three_bet: float,
        fold_to_cbet: float,
        aggression: float,
        wtsd: float,
        cbet_freq: float,
        check_raise_freq: float,
        overbet_freq: float,
        limp_freq: float,
        squeeze_freq: float,
        donk_freq: float,
    ) -> list[float]:
        """Convert player stats to normalized input vector.

        All inputs should be in [0, 1] range except aggression
        which is normalized by dividing by 10.
        """
        return [
            min(1.0, max(0.0, vpip)),
            min(1.0, max(0.0, pfr)),
            min(1.0, max(0.0, three_bet)),
            min(1.0, max(0.0, fold_to_cbet)),
            min(1.0, max(0.0, aggression / 10.0)),  # normalize AF
            min(1.0, max(0.0, wtsd)),
            min(1.0, max(0.0, cbet_freq)),
            min(1.0, max(0.0, check_raise_freq)),
            min(1.0, max(0.0, overbet_freq)),
            min(1.0, max(0.0, limp_freq)),
            min(1.0, max(0.0, squeeze_freq)),
            min(1.0, max(0.0, donk_freq)),
        ]
