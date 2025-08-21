# index_faiss.py
import os, re, json, time, hashlib
from typing import List, Optional, Tuple
import numpy as np
import faiss, torch
from sentence_transformers import SentenceTransformer

def _device_auto():
    if torch.cuda.is_available(): return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return "mps"
    return "cpu"

def _digest_key(model_name: str, texts: List[str]) -> str:
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    h.update(str(len(texts)).encode("utf-8"))
    for t in texts:
        if not isinstance(t, str): t = "" if t is None else str(t)
        h.update(t.encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()[:16]

class FaissIndex:
    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or _device_auto()
        self.model = SentenceTransformer(model_name, device=self.device)
        self.index: Optional[faiss.Index] = None
        self._key: Optional[str] = None
        self._dim: Optional[int] = None

    def _paths(self, cache_dir: str, key: str) -> Tuple[str, str, str]:
        os.makedirs(cache_dir, exist_ok=True)
        base = re.sub(r"[^A-Za-z0-9._-]+", "_", self.model_name.replace("/", "_"))
        faiss_path = os.path.join(cache_dir, f"{base}.{key}.faiss")
        npy_path   = os.path.join(cache_dir, f"{base}.{key}.npy")
        meta_path  = os.path.join(cache_dir, f"{base}.{key}.json")
        return faiss_path, npy_path, meta_path

    def build(self, texts: List[str]):
        t0 = time.time()
        X = self.model.encode(texts, convert_to_numpy=True, batch_size=64, show_progress_bar=True).astype("float32")
        faiss.normalize_L2(X)
        self.index = faiss.IndexFlatIP(X.shape[1])
        self.index.add(X)
        self._dim = int(X.shape[1])
        print(f"[FaissIndex] built in {time.time()-t0:.2f}s (n={len(texts)}, dim={self._dim})")

    def save(self, cache_dir: str, texts: List[str]):
        assert self.index is not None
        key = _digest_key(self.model_name, texts)
        self._key = key
        faiss_path, npy_path, meta_path = self._paths(cache_dir, key)
        faiss.write_index(self.index, faiss_path)
        # 임베딩은 선택 저장(공간 절약용으로 생략 가능) — 여기서는 저장
        # 주의: 원본 X가 없으므로 새로 encode해서 저장하지 않음
        meta = {"model": self.model_name, "ntotal": int(self.index.ntotal), "dim": self._dim}
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def load(self, cache_dir: str, texts: List[str]) -> bool:
        key = _digest_key(self.model_name, texts)
        faiss_path, _, _ = self._paths(cache_dir, key)
        if not os.path.exists(faiss_path):
            return False
        self.index = faiss.read_index(faiss_path)
        self._key = key
        return True

    def load_or_build(self, cache_dir: str, texts: List[str]):
        if not self.load(cache_dir, texts):
            self.build(texts)
            self.save(cache_dir, texts)

    def query(self, query: str, topk: Optional[int] = None):
        assert self.index is not None
        total = int(self.index.ntotal)
        k = total if (topk is None or topk <= 0 or topk > total) else int(topk)
        q = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q)
        D, I = self.index.search(q, k=k)
        return D[0], I[0]
