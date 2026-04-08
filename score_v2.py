import json
import glob
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def score_all():
    sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    single_files = sorted(glob.glob("results_*_single.json"))
    single = []
    for f in single_files:
        with open(f) as fh:
            single.extend(json.load(fh))

    if single:
        print(f"scoring {len(single)} single-needle results\n")
        for r in single:
            gold = r["answer"].lower().strip()
            resp = r["response"].lower().strip()
            r["recall"] = 1 if gold in resp else 0
            emb = sbert.encode([resp, gold])
            r["cosine_sim"] = float(cosine_similarity([emb[0]], [emb[1]])[0][0])

    multi_files = sorted(glob.glob("results_*_multi.json"))
    multi = []
    for f in multi_files:
        with open(f) as fh:
            multi.extend(json.load(fh))

    if multi:
        print(f"scoring {len(multi)} multi-needle results\n")
        for r in multi:
            resp = r["response"].lower().strip()
            found = sum(1 for a in r["answers"] if a.lower() in resp)
            r["recall"] = found / len(r["answers"])
            gold_concat = " ".join(r["answers"])
            emb = sbert.encode([resp, gold_concat])
            r["cosine_sim"] = float(cosine_similarity([emb[0]], [emb[1]])[0][0])

    reas_files = sorted(glob.glob("results_*_reasoning.json"))
    reas = []
    for f in reas_files:
        with open(f) as fh:
            reas.extend(json.load(fh))

    if reas:
        print(f"scoring {len(reas)} reasoning results\n")
        for r in reas:
            resp = r["response"].lower().strip()
            r["recall"] = 1 if r["answer"].lower() in resp else 0

    for task_name, data in [("SINGLE", single), ("MULTI", multi), ("REASONING", reas)]:
        if not data:
            continue
        print(f"\n{'='*60}")
        print(f"  {task_name} NEEDLE RESULTS")
        print(f"{'='*60}")
        models = sorted(set(r["model"] for r in data))
        ratios = sorted(set(r["ratio"] for r in data))

        header = f"{'model':<15} {'ratio':<8} {'recall':>8}"
        if task_name != "REASONING":
            header += f" {'cos_sim':>8}"
        header += f" {'n':>5}"
        print(header)
        print("-" * len(header))

        for m in models:
            for ratio in ratios:
                subset = [r for r in data if r["model"] == m and r["ratio"] == ratio]
                if not subset:
                    continue
                avg_recall = np.mean([r["recall"] for r in subset])
                line = f"{m:<15} {ratio:<8.2f} {avg_recall:>8.3f}"
                if task_name != "REASONING":
                    avg_cos = np.mean([r["cosine_sim"] for r in subset])
                    line += f" {avg_cos:>8.3f}"
                line += f" {len(subset):>5}"
                print(line)

    all_data = {"single": single, "multi": multi, "reasoning": reas}
    with open("scores_v2.json", "w") as f:
        json.dump(all_data, f, indent=2)

    all_summary = []
    for task_name, data in [("single", single), ("multi", multi), ("reasoning", reas)]:
        if not data:
            continue
        models = sorted(set(r["model"] for r in data))
        ratios = sorted(set(r["ratio"] for r in data))
        for m in models:
            for ratio in ratios:
                subset = [r for r in data if r["model"] == m and r["ratio"] == ratio]
                if not subset:
                    continue
                row = {"model": m, "ratio": ratio, "task": task_name, "recall": float(np.mean([r["recall"] for r in subset])), "n": len(subset)}
                if "cosine_sim" in subset[0]:
                    row["cosine_sim"] = float(np.mean([r["cosine_sim"] for r in subset]))
                all_summary.append(row)

    with open("summary_v2.json", "w") as f:
        json.dump(all_summary, f, indent=2)
    print(f"\nsaved scores_v2.json and summary_v2.json")

if __name__ == "__main__":
    score_all()
