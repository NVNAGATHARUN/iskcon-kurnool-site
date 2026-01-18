import requests
import json
from typing import List, Dict

BASE_URL = "https://vedicscriptures.github.io"

def fetch_chapters() -> List[Dict]:
    """Fetch metadata for all chapters."""
    url = f"{BASE_URL}/chapters"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return resp.json()

def fetch_slok(ch: int, sl: int) -> Dict:
    """Fetch one verse (slok) for given chapter and verse number."""
    url = f"{BASE_URL}/slok/{ch}/{sl}"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return resp.json()

def build_gita_dataset() -> List[Dict]:
    """Download all verses of Bhagavad Gita into a normalized list."""
    chapters = fetch_chapters()
    all_verses: List[Dict] = []

    for ch_info in chapters:
        chapter_number = ch_info["chapter_number"]
        verses_count = ch_info["verses_count"]

        print(f"Processing Chapter {chapter_number} with {verses_count} verses...")
        for sl in range(1, verses_count + 1):
            try:
                data = fetch_slok(chapter_number, sl)

                # Shape of /slok/:ch/:sl response (typical):
                # {
                #   "chapter": 2,
                #   "verse": 47,
                #   "slok": "...",
                #   "transliteration": "...",
                #   "meaning": {
                #       "en": "...",
                #       "hi": "..."
                #   },
                #   "commentary": { ... optional ... }
                # }
                verse_obj = {
                    "chapter": data.get("chapter", chapter_number),
                    "verse": data.get("verse", sl),
                    "slok": data.get("slok", ""),
                    "transliteration": data.get("transliteration", ""),
                    "meaning_en": (data.get("meaning") or {}).get("en", ""),
                    "meaning_hi": (data.get("meaning") or {}).get("hi", ""),
                    "commentary": data.get("commentary", {}),
                    # Extra field that will be handy for RAG
                    "ref": f"BG {chapter_number}.{sl}"
                }

                all_verses.append(verse_obj)
            except Exception as e:
                print(f"Error fetching {chapter_number}.{sl}: {e}")

    return all_verses

def main():
    verses = build_gita_dataset()
    print(f"Total verses collected: {len(verses)}")

    with open("gita_verses.json", "w", encoding="utf-8") as f:
        json.dump(verses, f, ensure_ascii=False, indent=2)

    print("Saved to gita_verses.json")

if __name__ == "__main__":
    main()
