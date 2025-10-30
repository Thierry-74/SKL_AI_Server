import torch
import torch.nn as nn
import numpy as np
from config import STATE_SIZE, NUM_MOVES, MODEL_PATH

# ------------------------------------------------------------
# 1️ Modèle PyTorch : simple MLP Policy + Value
# ------------------------------------------------------------
class TinyPolicyValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        hidden = 256
        self.backbone = nn.Sequential(
            nn.Linear(STATE_SIZE, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head_policy = nn.Linear(hidden, NUM_MOVES)
        self.head_value = nn.Linear(hidden, 1)

    def forward(self, x):
        z = self.backbone(x)
        policy_logits = self.head_policy(z)
        value = torch.tanh(self.head_value(z))
        return policy_logits, value


# ------------------------------------------------------------
# 2️ Classe d'inférence utilisée par FastAPI
# ------------------------------------------------------------
class InferenceEngine:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = TinyPolicyValueNet().to(self.device)
        self.model.eval()

        try:
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            print(f"✅ Modèle chargé depuis {MODEL_PATH}")
        except Exception as e:
            print(f"⚠️ Aucun modèle trouvé ({e}), génération aléatoire.")

    def infer(self, state_vec, legal_moves=None):
        if len(state_vec) != STATE_SIZE:
            raise ValueError(f"state size mismatch: expected {STATE_SIZE}, got {len(state_vec)}")

        if legal_moves is None:
            legal_moves = np.arange(NUM_MOVES)

        with torch.no_grad():
            x = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, value = self.model(x)
            logits = logits[0].cpu().numpy()
            value = float(value.item())

        # Masquage des coups illégaux
        mask = np.full(NUM_MOVES, -1e9, dtype=np.float32)
        mask[legal_moves] = 0.0
        masked_logits = logits + mask

        # Softmax
        exp = np.exp(masked_logits - masked_logits.max())
        policy = exp / (exp.sum() + 1e-12)

        return policy.astype(np.float32), value
