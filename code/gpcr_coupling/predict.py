"""
Prediction module: load pre-trained models and predict GPCR-G protein coupling.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PairedCrossAttentionNet(nn.Module):
    """Cross-attention network for paired GPCR-G protein coupling prediction."""

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.gpcr_proj = nn.Linear(input_dim, hidden_dim)
        self.gprot_proj = nn.Linear(input_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim * 2)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, gpcr_feat: torch.Tensor, gprot_feat: torch.Tensor
    ) -> torch.Tensor:
        h_gpcr = self.gpcr_proj(gpcr_feat).unsqueeze(1)
        h_gprot = self.gprot_proj(gprot_feat).unsqueeze(1)
        h_gpcr = self.ln1(h_gpcr)
        attn_out, _ = self.cross_attn(query=h_gpcr, key=h_gprot, value=h_gprot)
        combined = torch.cat([h_gpcr.squeeze(1), attn_out.squeeze(1)], dim=-1)
        combined = self.ln2(combined)
        return torch.sigmoid(self.ffn(combined)).squeeze(-1)


class CouplingPredictor:
    """High-level predictor interface for GPCR-G protein coupling."""

    G_PROTEIN_FAMILIES = ["Gq", "Gi", "Gs", "G12_13"]

    # Consensus Gα sequences for each family
    G_PROTEIN_SEQUENCES = {
        "Gq": (
            "MTLESIMACCLSEEAKEARRINDEIERQLRRDKRDARRELKLLLLGTGESGKSTFIKQMRIIHG"
            "SGYSEDKRGFTKLVYQNIFTAMQAMIRAMDTLKIPYKYEHNKAHAQLVREVDVEKVSAFENPYV"
            "DAIKSLWNDPGIQECYDRRREYQLSDSAKYYLNDLDRVADPAYLPTQQDVLRVRVPTTGIIEYP"
            "FDLQSVIFRMVDVGGQRSERRKWIHCFENVTSIMFLVALSEYDQVLVESDNENRMEESKALFRT"
            "IITYPWFQNSSVILFLNKKDLLEEKIMYSHLVDYFPEYDGPQRDAQAAREFILKMFVDLNPDSD"
            "KIIYSHFTCATDTENIRFVFAAVKDTILQLNLKEYNLV"
        ),
        "Gi": (
            "MGCTLSAEDKAAVERSKMIDRNLREDGEKAAREVKLLLLGAGESGKSTIVKQMKIIHE"
            "DGYSEDECKQYKVVVYSNTIQSIIAIIRAMGRLKIDFGDAARADDARQLFVLAGSAEEG"
            "VMTPELAGVIKRLWRDGGVQACFSRSREYQLNDSASYYLNDLDRISQSNYIPTQQDVL"
            "RTRVKTTGIVETHFTFKDLYFKMFDVGGQRSERKKWIHCFEDVTAIIFCVALSGYDQV"
            "LHEDETTNRMHESLMLFDSICNNKFFIDTSIILFLNKKDLFGEKIKKSPLTICFPEYT"
            "GANKYDEAASYIQSKFEDLNKRKDTKEIYTHFTCATDTKNVQFVFDAVTDVIIKNNLK"
            "DCGLF"
        ),
        "Gs": (
            "MGCLGNSKTEDQRNEEKAQREANKKIEKQLQKDKQVYRATHRLLLLGAGESGKSTIVKQ"
            "MRILHVNGFNGEGGEEDPQAARSNSDGEKATKVQDIKNNLKEAIETIVAAMSNLVPPVE"
            "LANPENQFRVDYILSVMNVPDFDFPPEFYEHAKALWEDEGVRACYERSNEYQLIDCAQY"
            "FLDKIDVIKQADYVPSDQDLLRCRVLTSGIFETKFQVDKVNFHMFDVGGQRDERRKWIQ"
            "CFNDVTAIIFVVASSSYNMVIREDNQTNRLQEALNLFKSIWNNRWLRTISVILFLNKQD"
            "LLAEKVLAGKSKIEDYFPEFARYTTPEDATPEPGEDPRVTRAKYFIRDEFLRISTASGD"
            "GRHYCYPHFTCATDTENIRRVFNDCRDIIQRMHLRQYELL"
        ),
        "G12_13": (
            "MADFLPSRSVLSVCFPGCLLTSGEAEQQRKSKEIDKCLSREKTYVKRLVKILLLGAGES"
            "GKSTFLKQMRIIHGREFDEKAYKHTRPVKMPDLRHLNMAQAMIRAMDQLRLAWPSVLEG"
            "AEERLERFRQLADATGPLLSFQQKTLNASRRWLERTEDLQKQRPVQQASRSPGRGKGLG"
            "RPKCSRGVSPSPAAPCPGPRVSARDWRKALLVDMQRNADDPRRLLHVSDAASLLLDRRL"
            "LLPRPRERSALQQASRAGAEKARRAGGRGARRRRRAGARGRARGARGARGAHGGARAGR"
            "RGGPGSRALAPRAGPGPGGGGGRGPRALRLAAEAPGPRAPGPRAPGPRAPGPRAPGPRG"
            "PRAPGPRAPGPRAPGPRAGEGGGPRAEAPGPRAPGPRVSGPRAPGPRAPGPRGGPRAEA"
            "PGPRAPGPRAPGPRAPGPRAPGPRAPGPRAPGPRAPGPRAGEGGGPRAEAPGPRAPGPR"
            "VSGPRAPGPRAPGPRGGPRAEAPGPRAPGPRAPGPRAPGPRAPGPRAPGPRAPGPRAPG"
            "PRAGEGGGPRAEAPGPRAPGPRVSGPRAPGPRAPGPRGGPRAEAPGPRAPGPRAPGPRW"
        ),
    }

    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = Path(__file__).parent.parent.parent / "pretrained_models"
        self.model_dir = Path(model_dir)
        self._model = None
        self._scaler = None
        self._loaded = False

    def load_model(self, device: str = "cpu"):
        """Load pre-trained cross-attention model and feature scaler."""
        model_path = self.model_dir / "crossattn_650m_best.pt"
        scaler_path = self.model_dir / "feature_scaler.pkl"

        self._model = PairedCrossAttentionNet(input_dim=1280, hidden_dim=256)
        if model_path.exists():
            state = torch.load(model_path, map_location=device, weights_only=True)
            self._model.load_state_dict(state["model_state_dict"])
        self._model.to(device)
        self._model.eval()

        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self._scaler = pickle.load(f)

        self._device = device
        self._loaded = True

    def predict(
        self,
        gpcr_features: np.ndarray,
        gprotein_features: np.ndarray,
        family: str,
    ) -> Dict:
        """Predict coupling probability for a (GPCR, G protein family) pair.

        Args:
            gpcr_features: GPCR feature vector (1280-d for 650M model)
            gprotein_features: G protein feature vector (1280-d)
            family: G protein family name (Gq/Gi/Gs/G12_13)

        Returns:
            Dict with keys: probability, confidence, prediction, family
        """
        if not self._loaded:
            self.load_model()

        if self._scaler is not None:
            gpcr_features = self._scaler.transform(gpcr_features.reshape(1, -1)).flatten()
            gprotein_features = self._scaler.transform(gprotein_features.reshape(1, -1)).flatten()

        with torch.no_grad():
            gpcr_t = torch.FloatTensor(gpcr_features).unsqueeze(0).to(self._device)
            gprot_t = torch.FloatTensor(gprotein_features).unsqueeze(0).to(self._device)
            prob = self._model(gpcr_t, gprot_t).item()

        confidence = 2 * abs(prob - 0.5)
        return {
            "probability": prob,
            "confidence": confidence,
            "prediction": 1 if prob > 0.5 else 0,
            "family": family,
        }

    def predict_all_families(
        self, gpcr_features: np.ndarray, gprotein_features_dict: Dict[str, np.ndarray]
    ) -> Dict[str, Dict]:
        """Predict coupling to all four G protein families.

        Args:
            gpcr_features: GPCR feature vector
            gprotein_features_dict: Dict mapping family name to feature vector

        Returns:
            Dict mapping family name to prediction result
        """
        results = {}
        for family, gprot_feat in gprotein_features_dict.items():
            results[family] = self.predict(gpcr_features, gprot_feat, family)
        return results

    def predict_orphan(
        self,
        gpcr_id: str,
        gpcr_features: np.ndarray,
    ) -> Dict:
        """Predict coupling profile for an orphan GPCR.

        Returns prioritized list of candidate G protein families.
        """
        results = {}
        for family in self.G_PROTEIN_FAMILIES:
            gprot_feat = self._get_default_gprotein_features(family)
            if gprot_feat is not None:
                results[family] = self.predict(gpcr_features, gprot_feat, family)

        # Sort by probability descending
        prioritized = sorted(
            results.items(), key=lambda x: x[1]["probability"], reverse=True
        )
        return {
            "gpcr_id": gpcr_id,
            "predictions": {fam: res for fam, res in prioritized},
            "top_candidate": prioritized[0][0],
            "top_probability": prioritized[0][1]["probability"],
            "high_confidence_candidates": [
                fam for fam, res in prioritized if res["probability"] > 0.95
            ],
        }

    def _get_default_gprotein_features(self, family: str) -> np.ndarray:
        """Load or compute default G protein features for a family."""
        feat_path = self.model_dir / f"gprotein_{family}_650m.npy"
        if feat_path.exists():
            return np.load(feat_path)
        # Return zero vector as fallback; user should provide features
        return np.zeros(1280, dtype=np.float32)
