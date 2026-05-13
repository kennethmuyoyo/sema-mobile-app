from .transformer_encoder import TransformerTagger
from .lstm_tagger import LSTMTagger

__all__ = ["TransformerTagger", "LSTMTagger", "build_model"]


def build_model(model_cfg: dict, input_dim: int, vocab_size: int):
    name = model_cfg["name"]
    aux = (model_cfg.get("aux_rvq") or {}) if isinstance(model_cfg.get("aux_rvq"), dict) else {}
    aux_codebook_size = int(aux["codebook_size"]) if aux.get("enabled") else 0

    if name == "transformer":
        return TransformerTagger(
            input_dim=input_dim,
            vocab_size=vocab_size,
            d_model=model_cfg["d_model"],
            n_heads=model_cfg["n_heads"],
            n_layers=model_cfg["n_layers"],
            ff_dim=model_cfg["ff_dim"],
            dropout=model_cfg["dropout"],
            input_norm=model_cfg.get("input_norm", True),
            max_len=model_cfg.get("max_len", 1024),
            aux_codebook_size=aux_codebook_size,
        )
    if name == "lstm":
        return LSTMTagger(
            input_dim=input_dim,
            vocab_size=vocab_size,
            hidden=model_cfg["hidden"],
            n_layers=model_cfg["n_layers"],
            bidirectional=model_cfg.get("bidirectional", True),
            dropout=model_cfg["dropout"],
            input_norm=model_cfg.get("input_norm", True),
        )
    raise ValueError(f"unknown model name: {name}")
