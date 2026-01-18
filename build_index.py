import json
import os
from typing import List, Dict

import faiss
from dotenv import load_dotenv
from google import genai

# ========== CONFIG ==========
EMBEDDING_MODEL = "text-embedding-004"  # current text embedding model name [web:103]
BATCH_SIZE = 64

DATA_PATH = "gita_verses.json"
INDEX_PATH = "gita_faiss.index"
META_PATH = "gita_meta.json"
# ============================


def load_data(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_text(verse: Dict) -> str:
    """Combine fields into one text string for embeddings."""
    slok = verse.get("slok", "") or ""
    translit = verse.get("transliteration", "") or ""
    meaning_en = verse.get("meaning_en", "") or ""
    ref = verse.get("ref", f"BG {verse.get('chapter')}.{verse.get('verse')}")

    parts = [
        ref,
        slok.strip(),
        translit.strip(),
        meaning_en.strip(),
    ]
    # Filter empty parts and join
    return " | ".join([p for p in parts if p])


def get_client():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set in environment or .env file")
    client = genai.Client(api_key=api_key)
    return client


def embed_batch(client, texts: List[str]) -> List[List[float]]:
    """Call Gemini embedding API for a batch of texts."""
    # google-genai SDK: client.models.embed_content [web:94][web:103]
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[{"parts": [{"text": t}]} for t in texts],
    )
    # result.embeddings is a list of objects with .values
    return [e.values for e in result.embeddings]


def main():
    print("Loading data...")
    verses = load_data(DATA_PATH)

    print(f"Total verses: {len(verses)}")
    client = get_client()

    embeddings: List[List[float]] = []
    meta: List[Dict] = []

    # ----- Build texts -----
    texts = [build_text(v) for v in verses]

    print("Generating embeddings in batches...")
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i : i + BATCH_SIZE]
        print(f"Batch {i} - {i + len(batch_texts)}")
        batch_embeds = embed_batch(client, batch_texts)
        embeddings.extend(batch_embeds)

        # meta aligned with embeddings
        for j, v in enumerate(verses[i : i + BATCH_SIZE]):
            meta.append(
                {
                    "id": i + j,
                    "chapter": v.get("chapter"),
                    "verse": v.get("verse"),
                    "ref": v.get("ref"),
                    "text": texts[i + j],
                }
            )

    if not embeddings:
        raise RuntimeError("No embeddings generated")

    dim = len(embeddings[0])
    print(f"Embedding dimension: {dim}")

    # Convert to FAISS index
    import numpy as np

    emb_array = np.array(embeddings, dtype="float32")
    index = faiss.IndexFlatL2(dim)  # L2 distance [web:44][web:99]
    index.add(emb_array)
    print(f"FAISS index size: {index.ntotal}")

    # Save index + metadata
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved index to {INDEX_PATH}")
    print(f"Saved metadata to {META_PATH}")


if __name__ == "__main__":
    main()
