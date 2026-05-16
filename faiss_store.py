import json
import os
import numpy as np
import faiss
from google import genai
from dataclasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()

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
    def __init__(self):
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
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

    def load(self, json_path: str, embeddings_path: str = "embeddings.npy"):
        """Load catalog + pre-computed embeddings into memory. No API calls."""
        with open(json_path,encoding="utf-8") as f:
            data = json.load(f)

        self.assessments = [self._to_assessment(item) for item in data]

        embeddings = np.load(embeddings_path).astype(np.float32)
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        print(f"Loaded {len(self.assessments)} assessments into memory.")

    def search(self, query: str, top_k: int = 10) -> list[Assessment]:
        if self.index is None:
            raise RuntimeError("Store not loaded. Call load() first.")

        result = self.client.models.embed_content(
            model="gemini-embedding-2",
            contents=[query],
        )
        vec = np.array([result.embeddings[0].values], dtype=np.float32)
        faiss.normalize_L2(vec)
        _, indices = self.index.search(vec, top_k)
        return [self.assessments[i] for i in indices[0] if i != -1]