import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass


@dataclass
class Assessment:
    entity_id: str
    name: str
    url: str
    description: str
    duration: str
    job_levels: list[str]
    languages: list[str]
    keys: list[str]
    remote: str
    adaptive: str


class FAISSStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.assessments: list[Assessment] = []

    def _to_assessment(self, item: dict) -> Assessment:
        return Assessment(
            entity_id=item.get("entity_id", ""),
            name=item.get("name", ""),
            url=item.get("link", ""),
            description=item.get("description", ""),
            duration=item.get("duration", ""),
            job_levels=item.get("job_levels", []),
            languages=item.get("languages", []),
            keys=item.get("keys", []),
            remote=item.get("remote", ""),
            adaptive=item.get("adaptive", ""),
        )

    def _make_text(self, a: Assessment) -> str:
        parts = [
            a.name,
            a.description,
            "Job levels: " + ", ".join(a.job_levels),
            "Keys: " + ", ".join(a.keys),
            "Duration: " + a.duration,
        ]
        return " | ".join(p for p in parts if p.strip(" |"))

    def load(self, json_path: str):
        """Ingest catalog from JSON into memory. Call once at startup."""
        with open(json_path, "r",encoding="utf-8") as f:
            data = json.load(f)

        self.assessments = [self._to_assessment(item) for item in data]
        texts = [self._make_text(a) for a in self.assessments]

        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        print(f"Loaded {len(self.assessments)} assessments into memory.")

    def search(self, query: str, top_k: int = 10) -> list[Assessment]:
        if self.index is None:
            raise RuntimeError("Store not loaded. Call load() first.")
        vec = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(vec)
        _, indices = self.index.search(vec, top_k)
        return [self.assessments[i] for i in indices[0] if i != -1]
    

if __name__=="__main__":
    store=FAISSStore()
    store.load("catalog.json")
    results = store.search("Java developer mid level")
    print(results)
