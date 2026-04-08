import json
import matplotlib.pyplot as plt
import numpy as np

def plot():
    with open("summary_v2.json") as f:
        data = json.load(f)

    colors = {"llama3-8b": "#e74c3c", "mistral-7b": "#3498db", "phi3-mini": "#2ecc71"}
    labels = {"llama3-8b": "Llama 3 8B (8K)", "mistral-7b": "Mistral 7B (32K)", "phi3-mini": "Phi-3-mini (128K)"}
    tasks = ["single", "multi", "reasoning"]
    task_titles = {"single": "Single Needle Recall", "multi": "Multi-Needle Recall (5 needles)", "reasoning": "Reasoning Accuracy"}

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax, task in zip(axes, tasks):
        task_data = [d for d in data if d["task"] == task]
        if not task_data:
            ax.set_title(f"{task_titles[task]} (no data)")
            continue

        models = sorted(set(d["model"] for d in task_data))
        for m in models:
            subset = sorted([d for d in task_data if d["model"] == m], key=lambda x: x["ratio"])
            ratios = [d["ratio"] * 100 for d in subset]
            recalls = [d["recall"] for d in subset]
            ax.plot(ratios, recalls, "o-", color=colors[m], label=labels[m], linewidth=2, markersize=7)

        ax.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5, label="90% threshold")
        ax.set_xlabel("Context Window Utilization (%)", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(task_titles[task], fontsize=13)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([25, 40, 50, 60, 75, 90, 95, 99])

    plt.suptitle("Context Window Utilization vs Performance", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("degradation_curves_v2.png", dpi=150, bbox_inches="tight")
    print("saved degradation_curves_v2.png")

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, task, title in [(ax1, "single", "Single Needle"), (ax2, "multi", "Multi-Needle")]:
        task_data = [d for d in data if d["task"] == task and "cosine_sim" in d]
        if not task_data:
            continue
        models = sorted(set(d["model"] for d in task_data))
        for m in models:
            subset = sorted([d for d in task_data if d["model"] == m], key=lambda x: x["ratio"])
            ratios = [d["ratio"] * 100 for d in subset]
            cosines = [d["cosine_sim"] for d in subset]
            ax.plot(ratios, cosines, "s-", color=colors[m], label=labels[m], linewidth=2, markersize=7)

        ax.set_xlabel("Context Window Utilization (%)", fontsize=11)
        ax.set_ylabel("SBERT Cosine Similarity", fontsize=11)
        ax.set_title(f"{title} - Semantic Similarity", fontsize=13)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([25, 40, 50, 60, 75, 90, 95, 99])

    plt.tight_layout()
    plt.savefig("cosine_curves_v2.png", dpi=150, bbox_inches="tight")
    print("saved cosine_curves_v2.png")

if __name__ == "__main__":
    plot()
