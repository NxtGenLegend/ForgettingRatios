import json
import os

FILLER_PATH = "filler_corpus.txt"
TARGET_CHARS = 2_000_000

def build_filler():
    if os.path.exists(FILLER_PATH):
        print(f"filler already exists at {FILLER_PATH}")
        return

    from datasets import load_dataset
    ds = load_dataset("emozilla/pg19", split="train", streaming=True)

    text = []
    total = 0
    for row in ds:
        t = row["text"]
        text.append(t)
        total += len(t)
        if total >= TARGET_CHARS:
            break
        if len(text) % 10 == 0:
            print(f"collected {total:,} chars from {len(text)} books")

    full = "\n\n".join(text)
    with open(FILLER_PATH, "w") as f:
        f.write(full)
    print(f"saved {len(full):,} chars to {FILLER_PATH}")

if __name__ == "__main__":
    build_filler()
