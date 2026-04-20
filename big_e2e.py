#!/usr/bin/env python3
"""
big_e2e.py — End-to-end pipeline: inference → scoring → plotting, with n trials per ratio.

Variance is introduced per task as follows (needle depth is always fixed at the midpoint,
where forgetting is most pronounced):

  single    — each trial randomly samples a subset of needles
  multi     — each trial uses a different random needle grouping (different seed)
  reasoning — each trial randomly samples a subset of reasoning pairs

This yields a mean ± std estimate of model performance at each context-window fill ratio.

Usage:
  python big_e2e.py --model llama3-8b --task all --n 5

Environment variable:
  TP_SIZE   tensor-parallel degree passed to vLLM (default 1)
"""

import json
import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from vllm import SamplingParams
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── Reuse constants and loaders from run_inference_v2 ────────────────────────
from run_inference_v2 import (
    RATIOS,
    SYSTEM_SINGLE, SYSTEM_MULTI, SYSTEM_REASONING,
    load_needles, load_filler, load_reasoning_pairs,
)
from model_utils import MODEL_REGISTRY, load_model, preprocess_messages, postprocess_response

# ── Filler helper ─────────────────────────────────────────────────────────────

def get_filler_tokens(tokenizer, filler, n_tokens):
    """Return exactly n_tokens tokens from filler, looping the corpus if needed."""
    tokens = tokenizer.encode(filler, add_special_tokens=False)
    if len(tokens) >= n_tokens:
        return tokens[:n_tokens]
    result = []
    while len(result) < n_tokens:
        result.extend(tokens)
    return result[:n_tokens]


# ── Prompt builders ───────────────────────────────────────────────────────────

def build_single_prompt(tokenizer, filler, needle, question, target_tokens, model_key):
    """Single-needle prompt; needle is always placed at the midpoint of the filler."""
    needle_toks = len(tokenizer.encode(needle, add_special_tokens=False))
    question_str = f"\n\nQuestion: {question}\nAnswer:"
    q_toks   = len(tokenizer.encode(question_str,  add_special_tokens=False))
    sys_toks = len(tokenizer.encode(SYSTEM_SINGLE, add_special_tokens=False))
    overhead = 20

    filler_budget = max(100, target_tokens - needle_toks - q_toks - sys_toks - overhead)
    filler_tokens = get_filler_tokens(tokenizer, filler, filler_budget)

    mid   = len(filler_tokens) // 2
    left  = tokenizer.decode(filler_tokens[:mid])
    right = tokenizer.decode(filler_tokens[mid:])
    context = left + "\n" + needle + "\n" + right + question_str

    messages = [
        {"role": "system", "content": SYSTEM_SINGLE},
        {"role": "user",   "content": context},
    ]
    messages = preprocess_messages(messages, model_key)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_multi_prompt(tokenizer, filler, group, target_tokens, model_key):
    """Multi-needle prompt; needles are spread evenly through the middle third."""
    questions    = "\n".join(f"{i+1}. {n['question']}" for i, n in enumerate(group))
    question_str = (
        f"\n\nAnswer each question below based only on the context above. "
        f"Number your answers.\n{questions}\n\nAnswers:"
    )

    needle_texts      = [n["needle"] for n in group]
    total_needle_toks = sum(len(tokenizer.encode(t, add_special_tokens=False)) for t in needle_texts)
    q_toks   = len(tokenizer.encode(question_str, add_special_tokens=False))
    sys_toks = len(tokenizer.encode(SYSTEM_MULTI, add_special_tokens=False))
    overhead = 20

    filler_budget = max(100, target_tokens - total_needle_toks - q_toks - sys_toks - overhead)
    filler_tokens = get_filler_tokens(tokenizer, filler, filler_budget)

    third        = len(filler_tokens) // 3
    left_tokens  = filler_tokens[:third]
    mid_tokens   = filler_tokens[third:2*third]
    right_tokens = filler_tokens[2*third:]

    n_needles = len(needle_texts)
    seg_len   = max(1, len(mid_tokens) // (n_needles + 1))

    mid_parts = []
    for i in range(n_needles):
        start = i * seg_len
        end   = (i + 1) * seg_len
        mid_parts.append(tokenizer.decode(mid_tokens[start:end]))
        mid_parts.append("\n" + needle_texts[i] + "\n")
    mid_parts.append(tokenizer.decode(mid_tokens[n_needles * seg_len:]))

    context = (
        tokenizer.decode(left_tokens)
        + "".join(mid_parts)
        + tokenizer.decode(right_tokens)
        + question_str
    )
    messages = [
        {"role": "system", "content": SYSTEM_MULTI},
        {"role": "user",   "content": context},
    ]
    messages = preprocess_messages(messages, model_key)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_reasoning_prompt(tokenizer, filler, needle_a, needle_b, question, target_tokens, model_key):
    """Two-needle reasoning prompt; needles fixed at 1/3 and 2/3 of the filler."""
    needle_a_toks = len(tokenizer.encode(needle_a, add_special_tokens=False))
    needle_b_toks = len(tokenizer.encode(needle_b, add_special_tokens=False))
    question_str  = f"\n\nQuestion: {question}\nAnswer:"
    q_toks   = len(tokenizer.encode(question_str,     add_special_tokens=False))
    sys_toks = len(tokenizer.encode(SYSTEM_REASONING, add_special_tokens=False))
    overhead = 20

    filler_budget = max(100, target_tokens - needle_a_toks - needle_b_toks - q_toks - sys_toks - overhead)
    filler_tokens = get_filler_tokens(tokenizer, filler, filler_budget)

    third     = len(filler_tokens) // 3
    mid_start = third
    mid_end   = 2 * third
    mid       = (mid_start + mid_end) // 2

    p1  = tokenizer.decode(filler_tokens[:mid_start])
    p2a = tokenizer.decode(filler_tokens[mid_start:mid])
    p2b = tokenizer.decode(filler_tokens[mid:mid_end])
    p3  = tokenizer.decode(filler_tokens[mid_end:])

    context = p1 + "\n" + needle_a + "\n" + p2a + "\n" + needle_b + "\n" + p2b + p3 + question_str
    messages = [
        {"role": "system", "content": SYSTEM_REASONING},
        {"role": "user",   "content": context},
    ]
    messages = preprocess_messages(messages, model_key)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# ── Batched n-trial runners ───────────────────────────────────────────────────
# All trials for a given ratio are batched into a single llm.generate call.
# Each result dict carries a "trial" field for downstream aggregation.

def run_single_ntimes(llm, tokenizer, cfg, model_key, needles, filler,
                      ratio, n_trials, base_seed, n_samples):
    """
    Single-needle task for one ratio, n_trials times.
    Variance source: each trial draws a different random subset of n_samples needles.
    Needle is always placed at the midpoint (middle of the context window).
    """
    target = int(ratio * cfg["ctx"])
    params = SamplingParams(temperature=0, top_p=1.0, max_tokens=100)

    prompts, meta = [], []
    for trial in range(n_trials):
        rng    = random.Random(base_seed + trial * 997 + int(ratio * 1000))
        subset = rng.sample(needles, min(n_samples, len(needles)))

        for i, n in enumerate(subset):
            p = build_single_prompt(tokenizer, filler, n["needle"], n["question"], target, model_key)
            prompts.append(p)
            meta.append({
                "trial": trial, "needle_idx": i,
                "ratio": ratio,
                "question": n["question"], "answer": n["answer"], "needle": n["needle"],
                "model": model_key, "task": "single",
            })

    outputs = llm.generate(prompts, params)
    for m, o in zip(meta, outputs):
        m["response"] = postprocess_response(o.outputs[0].text.strip(), model_key)
    return meta


def run_multi_ntimes(llm, tokenizer, cfg, model_key, needles, filler,
                     ratio, n_trials, base_seed, group_size=5, n_groups=20):
    """
    Multi-needle task for one ratio, n_trials times.
    Variance source: each trial uses a different random grouping of needles.
    """
    target = int(ratio * cfg["ctx"])
    params = SamplingParams(temperature=0, top_p=1.0, max_tokens=500)

    prompts, meta = [], []
    for trial in range(n_trials):
        seed = base_seed + trial * 997 + int(ratio * 1000)
        rng  = random.Random(seed)
        idxs = list(range(len(needles)))
        rng.shuffle(idxs)

        groups = []
        for i in range(0, min(n_groups * group_size, len(idxs)), group_size):
            if i + group_size <= len(idxs):
                groups.append([needles[idxs[j]] for j in range(i, i + group_size)])

        for gi, group in enumerate(groups):
            p = build_multi_prompt(tokenizer, filler, group, target, model_key)
            prompts.append(p)
            meta.append({
                "trial": trial, "group_idx": gi,
                "ratio": ratio,
                "questions": [n["question"] for n in group],
                "answers":   [n["answer"]   for n in group],
                "needles":   [n["needle"]   for n in group],
                "model": model_key, "task": "multi",
            })

    outputs = llm.generate(prompts, params)
    for m, o in zip(meta, outputs):
        m["response"] = postprocess_response(o.outputs[0].text.strip(), model_key)
    return meta


def run_reasoning_ntimes(llm, tokenizer, cfg, model_key, needles, filler,
                         ratio, n_trials, base_seed, n_pairs):
    """
    Reasoning task for one ratio, n_trials times.
    Variance source: each trial draws a different random subset of n_pairs reasoning pairs.
    Needle positions are fixed (1/3 and 2/3 of the filler).
    """
    all_pairs = load_reasoning_pairs()
    target    = int(ratio * cfg["ctx"])
    params    = SamplingParams(temperature=0, top_p=1.0, max_tokens=150)

    prompts, meta = [], []
    for trial in range(n_trials):
        rng   = random.Random(base_seed + trial * 997 + int(ratio * 1000))
        pairs = rng.sample(all_pairs, min(n_pairs, len(all_pairs)))

        for pi, pair in enumerate(pairs):
            na = needles[pair["a_idx"]]
            nb = needles[pair["b_idx"]]
            p  = build_reasoning_prompt(
                tokenizer, filler, na["needle"], nb["needle"], pair["question"], target, model_key
            )
            prompts.append(p)
            meta.append({
                "trial": trial, "pair_idx": pi,
                "ratio": ratio,
                "question": pair["question"], "answer": pair["answer"],
                "needle_a": na["needle"],     "needle_b": nb["needle"],
                "model": model_key, "task": "reasoning",
            })

    outputs = llm.generate(prompts, params)
    for m, o in zip(meta, outputs):
        m["response"] = postprocess_response(o.outputs[0].text.strip(), model_key)
    return meta


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_results(results, sbert, task):
    """In-place scoring: adds 'recall' (and 'cosine_sim' for single/multi)."""
    for r in results:
        resp = r["response"].lower().strip()
        if task == "single":
            gold        = r["answer"].lower().strip()
            r["recall"] = 1 if gold in resp else 0
            emb         = sbert.encode([resp, gold])
            r["cosine_sim"] = float(cosine_similarity([emb[0]], [emb[1]])[0][0])
        elif task == "multi":
            found       = sum(1 for a in r["answers"] if a.lower() in resp)
            r["recall"] = found / len(r["answers"])
            gold_concat = " ".join(r["answers"])
            emb         = sbert.encode([resp, gold_concat])
            r["cosine_sim"] = float(cosine_similarity([emb[0]], [emb[1]])[0][0])
        elif task == "reasoning":
            r["recall"] = 1 if r["answer"].lower() in resp else 0


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_trials(results, model_key, task, ratio, n_trials):
    """
    Compute per-trial mean recall (and cosine_sim), then return the
    cross-trial mean ± std as a summary dict.
    """
    trial_recalls, trial_cosines = [], []
    for trial in range(n_trials):
        subset = [r for r in results if r["trial"] == trial]
        if not subset:
            continue
        trial_recalls.append(float(np.mean([r["recall"] for r in subset])))
        if "cosine_sim" in subset[0]:
            trial_cosines.append(float(np.mean([r["cosine_sim"] for r in subset])))

    n = len(trial_recalls)
    row = {
        "model":          model_key,
        "task":           task,
        "ratio":          ratio,
        "n_trials":       n,
        "recall_mean":    float(np.mean(trial_recalls)),
        "recall_std":     float(np.std(trial_recalls, ddof=1)) if n > 1 else 0.0,
        "trial_recalls":  trial_recalls,
    }
    if trial_cosines:
        row["cosine_mean"]   = float(np.mean(trial_cosines))
        row["cosine_std"]    = float(np.std(trial_cosines, ddof=1)) if n > 1 else 0.0
        row["trial_cosines"] = trial_cosines
    return row


# ── Plotting (adapted from plot_v2.py, with ± std shading) ───────────────────

COLORS = {
    # Original models
    "llama3-8b":       "#e74c3c",
    "mistral-7b":      "#3498db",
    "phi3-mini":       "#2ecc71",
    # Qwen2.5 — orange family
    "qwen25-7b-128k":  "#e67e22",
    "qwen25-7b-1m":    "#f39c12",
    "qwen25-14b-128k": "#d35400",
    "qwen25-14b-1m":   "#c0392b",
    # Gemma — purple family
    "gemma2-27b":      "#9b59b6",
    "gemma3-27b":      "#6c3483",
    # OLMo — teal family
    "olmo2-32b":       "#1abc9c",
    "olmo3-32b":       "#148f77",
}
LABELS = {
    # Original models
    "llama3-8b":       "Llama 3 8B (8K)",
    "mistral-7b":      "Mistral 7B (32K)",
    "phi3-mini":       "Phi-3-mini (128K)",
    # Qwen2.5
    "qwen25-7b-128k":  "Qwen2.5-7B (128K)",
    "qwen25-7b-1m":    "Qwen2.5-7B-1M (1M)",
    "qwen25-14b-128k": "Qwen2.5-14B (128K)",
    "qwen25-14b-1m":   "Qwen2.5-14B-1M (1M)",
    # Gemma
    "gemma2-27b":      "Gemma 2-27B (8K)",
    "gemma3-27b":      "Gemma 3-27B (128K)",
    # OLMo
    "olmo2-32b":       "OLMo 2-32B (4K)",
    "olmo3-32b":       "OLMo 3.1-32B (65K)",
}
TASK_TITLES = {
    "single":    "Single Needle Recall",
    "multi":     "Multi-Needle Recall (5 needles)",
    "reasoning": "Reasoning Accuracy",
}
XTICKS = [25, 40, 50, 60, 75, 90, 95, 99]


def _plot_task_ax(ax, task, summary):
    task_data = [d for d in summary if d["task"] == task]
    if not task_data:
        ax.set_title(f"{TASK_TITLES[task]} (no data)")
        return

    for m in sorted(set(d["model"] for d in task_data)):
        subset = sorted([d for d in task_data if d["model"] == m], key=lambda x: x["ratio"])
        ratios = [d["ratio"] * 100     for d in subset]
        means  = [d["recall_mean"]     for d in subset]
        stds   = [d["recall_std"]      for d in subset]
        c      = COLORS.get(m, "#888888")

        ax.plot(ratios, means, "o-", color=c, label=LABELS.get(m, m), linewidth=2, markersize=7)
        ax.fill_between(
            ratios,
            [mu - s for mu, s in zip(means, stds)],
            [mu + s for mu, s in zip(means, stds)],
            color=c, alpha=0.18,
        )

    ax.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5, label="90% threshold")
    ax.set_xlabel("Context Window Utilization (%)", fontsize=11)
    ax.set_ylabel("Accuracy (mean ± std)", fontsize=11)
    ax.set_title(TASK_TITLES[task], fontsize=13)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(XTICKS)


def plot_e2e(summary, out_prefix="e2e"):
    # ── Recall / accuracy curves ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, task in zip(axes, ["single", "multi", "reasoning"]):
        _plot_task_ax(ax, task, summary)

    plt.suptitle(
        "Context Window Utilization vs Performance (mean ± std across trials)",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fname = f"{out_prefix}_degradation_curves.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {fname}")

    # ── Cosine similarity curves (single + multi only) ────────────────────────
    cosine_tasks = [
        d for d in summary
        if d.get("cosine_mean") is not None and d["task"] in ("single", "multi")
    ]
    if cosine_tasks:
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
        for ax, task, title in zip(axes2,
                                   ["single",       "multi"],
                                   ["Single Needle", "Multi-Needle"]):
            td = [d for d in cosine_tasks if d["task"] == task]
            if not td:
                continue
            for m in sorted(set(d["model"] for d in td)):
                subset = sorted([d for d in td if d["model"] == m], key=lambda x: x["ratio"])
                ratios = [d["ratio"] * 100  for d in subset]
                means  = [d["cosine_mean"]  for d in subset]
                stds   = [d["cosine_std"]   for d in subset]
                c      = COLORS.get(m, "#888888")

                ax.plot(ratios, means, "s-", color=c, label=LABELS.get(m, m), linewidth=2, markersize=7)
                ax.fill_between(
                    ratios,
                    [mu - s for mu, s in zip(means, stds)],
                    [mu + s for mu, s in zip(means, stds)],
                    color=c, alpha=0.18,
                )

            ax.set_xlabel("Context Window Utilization (%)", fontsize=11)
            ax.set_ylabel("SBERT Cosine Similarity (mean ± std)", fontsize=11)
            ax.set_title(f"{title} — Semantic Similarity", fontsize=13)
            ax.legend(fontsize=9)
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(XTICKS)

        plt.tight_layout()
        fname2 = f"{out_prefix}_cosine_curves.png"
        plt.savefig(fname2, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"saved {fname2}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="End-to-end inference → scoring → plotting with n trials per ratio."
    )
    p.add_argument("--model",      required=True, choices=sorted(MODEL_REGISTRY.keys()))
    p.add_argument("--task",       default="all", choices=["single", "multi", "reasoning", "all"])
    p.add_argument("--n",          type=int, default=5,  help="Trials per ratio")
    p.add_argument("--n-single",   type=int, default=50, help="Needles sampled per single trial")
    p.add_argument("--n-pairs",    type=int, default=20, help="Pairs sampled per reasoning trial")
    p.add_argument("--base-seed",  type=int, default=42, help="Base random seed")
    p.add_argument("--out-prefix", default="e2e",        help="Output filename prefix")
    args = p.parse_args()

    cfg = MODEL_REGISTRY[args.model]
    print(f"loading {cfg['hf']}...")
    tokenizer, llm = load_model(args.model)

    print("loading SBERT scorer...")
    sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    needles = load_needles()
    filler  = load_filler()

    tasks = ["single", "multi", "reasoning"] if args.task == "all" else [args.task]

    all_raw_results: list = []
    summary:         list = []

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"  {task.upper()} | {args.model} | {args.n} trials × {len(RATIOS)} ratios")
        if task == "single":
            print(f"  ({args.n_single} needles sampled per trial)")
        elif task == "reasoning":
            print(f"  ({args.n_pairs} pairs sampled per trial)")
        print(f"{'='*60}")

        for ratio in RATIOS:
            print(f"\n--- {task} | ratio={ratio} ---")

            if task == "single":
                results = run_single_ntimes(
                    llm, tokenizer, cfg, args.model, needles, filler,
                    ratio, args.n, args.base_seed, args.n_single,
                )
            elif task == "multi":
                results = run_multi_ntimes(
                    llm, tokenizer, cfg, args.model, needles, filler,
                    ratio, args.n, args.base_seed,
                )
            elif task == "reasoning":
                results = run_reasoning_ntimes(
                    llm, tokenizer, cfg, args.model, needles, filler,
                    ratio, args.n, args.base_seed, args.n_pairs,
                )

            score_results(results, sbert, task)

            row = aggregate_trials(results, args.model, task, ratio, args.n)
            summary.append(row)

            print(f"  trial recalls : {[f'{r:.3f}' for r in row['trial_recalls']]}")
            print(f"  mean={row['recall_mean']:.3f}  std={row['recall_std']:.3f}")

            all_raw_results.extend(results)

    # ── Persist results ───────────────────────────────────────────────────────
    raw_path = f"{args.out_prefix}_{args.model}_raw.json"
    with open(raw_path, "w") as f:
        json.dump(all_raw_results, f, indent=2)
    print(f"\nsaved {len(all_raw_results)} raw results → {raw_path}")

    summary_path = f"{args.out_prefix}_{args.model}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved summary → {summary_path}")

    # ── Final table ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    hdr = f"{'task':<12} {'ratio':<8} {'mean':>8} {'std':>8} {'trials':>7}"
    print(hdr)
    print("-" * len(hdr))
    for row in summary:
        print(
            f"{row['task']:<12} {row['ratio']:<8.2f} "
            f"{row['recall_mean']:>8.3f} {row['recall_std']:>8.3f} "
            f"{row['n_trials']:>7}"
        )

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_e2e(summary, out_prefix=args.out_prefix)


if __name__ == "__main__":
    main()
