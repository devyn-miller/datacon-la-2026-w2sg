"""Model loading with offline fallback for network-free testing."""
import hashlib

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertConfig, BertModel

_HASH_VOCAB_SIZE = 8000
_PAD, _CLS, _SEP, _UNK = 0, 1, 2, 3
_SPECIAL_TOKENS = 4


class HashingTokenizer:
    """Hash-based tokenizer for offline mode."""

    def __init__(self, vocab_size: int = _HASH_VOCAB_SIZE, max_length: int = 256):
        self.vocab_size = vocab_size
        self.max_length = max_length

    def _hash(self, token: str) -> int:
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        return _SPECIAL_TOKENS + (int.from_bytes(digest[:4], "little") % (self.vocab_size - _SPECIAL_TOKENS))

    def __call__(self, text_a: str, text_b: str) -> dict:
        ids = [_CLS] + [self._hash(t) for t in text_a.lower().split()] + [_SEP]
        ids += [self._hash(t) for t in text_b.lower().split()] + [_SEP]
        ids = ids[: self.max_length]
        token_type = [0] * (len(text_a.split()) + 2) + [1] * (len(ids) - (len(text_a.split()) + 2))
        token_type = token_type[: self.max_length]
        pad_len = self.max_length - len(ids)
        attn = [1] * len(ids) + [0] * pad_len
        ids = ids + [_PAD] * pad_len
        token_type = token_type + [0] * pad_len
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.tensor([attn]),
            "token_type_ids": torch.tensor([token_type]),
        }

    def batch(self, pairs: list[tuple[str, str]]) -> dict:
        encoded = [self(a, b) for a, b in pairs]
        return {
            k: torch.cat([e[k] for e in encoded], dim=0) for k in ("input_ids", "attention_mask", "token_type_ids")
        }


def _tiny_bert_config() -> BertConfig:
    return BertConfig(
        vocab_size=_HASH_VOCAB_SIZE,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=256,
        pad_token_id=_PAD,
    )


class OfflineCrossEncoder(nn.Module):
    """Offline cross-encoder for testing."""

    def __init__(self, max_length: int = 256):
        super().__init__()
        self.tokenizer = HashingTokenizer(max_length=max_length)
        cfg = _tiny_bert_config()
        cfg.num_labels = 1
        self.encoder = BertModel(cfg)
        self.head = nn.Linear(cfg.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls = out.last_hidden_state[:, 0]
        return self.head(cls).squeeze(-1)

    def encode_pairs(self, pairs: list[tuple[str, str]]) -> dict:
        return self.tokenizer.batch(pairs)

    @torch.no_grad()
    def predict(self, pairs: list[tuple[str, str]], batch_size: int = 32) -> list[float]:
        self.eval()
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch = self.tokenizer.batch(pairs[i : i + batch_size])
            scores.extend(self(**batch).tolist())
        return scores


class OfflineBiEncoder:
    """Offline bi-encoder for testing."""

    def __init__(self):
        self.tokenizer = HashingTokenizer(max_length=128)
        self.model = BertModel(_tiny_bert_config())
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts, batch_size: int = 64, show_progress_bar: bool = False, normalize_embeddings: bool = True):
        single = isinstance(texts, str)
        texts = [texts] if single else texts
        embs = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            enc = self.tokenizer.batch([(t, "") for t in chunk])
            out = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
            if normalize_embeddings:
                pooled = torch.nn.functional.normalize(pooled, dim=-1)
            embs.append(pooled)
        result = torch.cat(embs, dim=0).numpy()
        return result[0] if single else result


def load_dense_encoder(model_name: str, offline: bool):
    if offline:
        return OfflineBiEncoder()
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def load_cross_encoder_scorer(model_name: str, max_length: int, offline: bool):
    if offline:
        return OfflineCrossEncoder(max_length=max_length)
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, max_length=max_length)


class HfStudent(nn.Module):

    def __init__(self, model_name: str, max_length: int):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
        self.max_length = max_length

    def encode_pairs(self, pairs: list[tuple[str, str]]):
        texts_a, texts_b = zip(*pairs)
        enc = self.tokenizer(
            list(texts_a), list(texts_b), padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        return enc

    def forward(self, **encoded):
        return self.encoder(**encoded).logits.squeeze(-1)

    @torch.no_grad()
    def predict(self, pairs: list[tuple[str, str]], batch_size: int = 32) -> list[float]:
        self.eval()
        scores = []
        for i in range(0, len(pairs), batch_size):
            enc = self.encode_pairs(pairs[i : i + batch_size])
            scores.extend(self(**enc).tolist())
        return scores


def load_student(model_name: str, max_length: int, offline: bool):
    if offline:
        return OfflineCrossEncoder(max_length=max_length)
    return HfStudent(model_name, max_length)
