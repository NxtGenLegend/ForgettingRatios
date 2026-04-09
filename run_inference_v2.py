import json
import os
import argparse
import random
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODELS = {
    "llama3-8b": {"hf": "meta-llama/Meta-Llama-3-8B-Instruct", "ctx": 8192},
    "mistral-7b": {"hf": "mistralai/Mistral-7B-Instruct-v0.3", "ctx": 32768},
    "phi3-mini": {"hf": "microsoft/Phi-3-mini-128k-instruct", "ctx": 131072},
}

RATIOS = [0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99]

SYSTEM_SINGLE = "You are a helpful assistant. Answer the question based only on the provided context. Be concise."
SYSTEM_MULTI = "You are a helpful assistant. Answer each question based only on the provided context. Be concise. Number your answers."
SYSTEM_REASONING = "You are a helpful assistant. Answer the question based only on the provided context. Give a short answer."

def load_needles():
    with open("needles_new.json") as f:
        return json.load(f)

def load_filler():
    with open("filler_corpus.txt") as f:
        return f.read()

def load_reasoning_pairs():
    with open("reasoning_pairs_new.json") as f:
        return json.load(f)

def build_single_prompt(tokenizer, filler, needle, question, target_tokens):
    needle_toks = len(tokenizer.encode(needle, add_special_tokens=False))
    question_str = f"\n\nQuestion: {question}\nAnswer:"
    q_toks = len(tokenizer.encode(question_str, add_special_tokens=False))
    sys_toks = len(tokenizer.encode(SYSTEM_SINGLE, add_special_tokens=False))
    overhead = 20

    filler_budget = target_tokens - needle_toks - q_toks - sys_toks - overhead
    if filler_budget < 100:
        filler_budget = 100

    filler_tokens = tokenizer.encode(filler, add_special_tokens=False)[:filler_budget]
    mid = len(filler_tokens) // 2
    left = tokenizer.decode(filler_tokens[:mid])
    right = tokenizer.decode(filler_tokens[mid:])

    context = left + "\n" + needle + "\n" + right + question_str
    messages = [
        {"role": "system", "content": SYSTEM_SINGLE},
        {"role": "user", "content": context},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def run_single(llm, tokenizer, cfg, model_key, needles, filler):
    params = SamplingParams(temperature=0, top_p=1.0, max_tokens=100)
    results = []

    for ratio in RATIOS:
        target = int(ratio * cfg["ctx"])
        print(f"\n--- single | {model_key} | ratio={ratio} | target={target} ---")

        prompts, meta = [], []
        for i, n in enumerate(needles):
            p = build_single_prompt(tokenizer, filler, n["needle"], n["question"], target)
            prompts.append(p)
            meta.append({"needle_idx": i, "ratio": ratio, "question": n["question"], "answer": n["answer"], "needle": n["needle"]})

        outputs = llm.generate(prompts, params)
        for m, o in zip(meta, outputs):
            resp = o.outputs[0].text.strip()
            m["response"] = resp
            m["model"] = model_key
            m["task"] = "single"
            results.append(m)

        correct = sum(1 for m in meta if m["answer"].lower() in m["response"].lower())
        print(f"  recall: {correct}/{len(meta)}")

    return results

def make_multi_groups(needles, group_size=5, n_groups=20):
    random.seed(42)
    idxs = list(range(len(needles)))
    random.shuffle(idxs)
    groups = []
    for i in range(0, min(n_groups * group_size, len(idxs)), group_size):
        if i + group_size <= len(idxs):
            groups.append([needles[idxs[j]] for j in range(i, i + group_size)])
    return groups

def build_multi_prompt(tokenizer, filler, group, target_tokens):
    questions = "\n".join([f"{i+1}. {n['question']}" for i, n in enumerate(group)])
    question_str = f"\n\nAnswer each question below based only on the context above. Number your answers.\n{questions}\n\nAnswers:"

    needle_texts = [n["needle"] for n in group]
    total_needle_toks = sum(len(tokenizer.encode(n, add_special_tokens=False)) for n in needle_texts)
    q_toks = len(tokenizer.encode(question_str, add_special_tokens=False))
    sys_toks = len(tokenizer.encode(SYSTEM_MULTI, add_special_tokens=False))
    overhead = 20

    filler_budget = target_tokens - total_needle_toks - q_toks - sys_toks - overhead
    if filler_budget < 100:
        filler_budget = 100

    filler_tokens = tokenizer.encode(filler, add_special_tokens=False)[:filler_budget]

    third = len(filler_tokens) // 3
    left_tokens = filler_tokens[:third]
    mid_tokens = filler_tokens[third:2*third]
    right_tokens = filler_tokens[2*third:]

    n_needles = len(needle_texts)
    seg_len = len(mid_tokens) // (n_needles + 1)

    mid_parts = []
    for i in range(n_needles):
        start = i * seg_len
        end = (i + 1) * seg_len
        mid_parts.append(tokenizer.decode(mid_tokens[start:end]))
        mid_parts.append("\n" + needle_texts[i] + "\n")
    mid_parts.append(tokenizer.decode(mid_tokens[n_needles * seg_len:]))

    context = tokenizer.decode(left_tokens) + "".join(mid_parts) + tokenizer.decode(right_tokens) + question_str

    messages = [
        {"role": "system", "content": SYSTEM_MULTI},
        {"role": "user", "content": context},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def run_multi(llm, tokenizer, cfg, model_key, needles, filler):
    params = SamplingParams(temperature=0, top_p=1.0, max_tokens=500)
    groups = make_multi_groups(needles)
    results = []

    for ratio in RATIOS:
        target = int(ratio * cfg["ctx"])
        print(f"\n--- multi | {model_key} | ratio={ratio} | target={target} ---")

        prompts, meta = [], []
        for gi, group in enumerate(groups):
            p = build_multi_prompt(tokenizer, filler, group, target)
            prompts.append(p)
            meta.append({
                "group_idx": gi,
                "ratio": ratio,
                "questions": [n["question"] for n in group],
                "answers": [n["answer"] for n in group],
                "needles": [n["needle"] for n in group],
            })

        outputs = llm.generate(prompts, params)
        total_found = 0
        total_possible = 0
        for m, o in zip(meta, outputs):
            resp = o.outputs[0].text.strip()
            m["response"] = resp
            m["model"] = model_key
            m["task"] = "multi"
            found = sum(1 for a in m["answers"] if a.lower() in resp.lower())
            m["found"] = found
            m["total"] = len(m["answers"])
            m["recall"] = found / len(m["answers"])
            total_found += found
            total_possible += len(m["answers"])
            results.append(m)

        print(f"  recall: {total_found}/{total_possible} = {total_found/total_possible:.3f}")

    return results

def build_reasoning_prompt(tokenizer, filler, needle_a, needle_b, question, target_tokens):
    needle_a_toks = len(tokenizer.encode(needle_a, add_special_tokens=False))
    needle_b_toks = len(tokenizer.encode(needle_b, add_special_tokens=False))
    question_str = f"\n\nQuestion: {question}\nAnswer:"
    q_toks = len(tokenizer.encode(question_str, add_special_tokens=False))
    sys_toks = len(tokenizer.encode(SYSTEM_REASONING, add_special_tokens=False))
    overhead = 20

    filler_budget = target_tokens - needle_a_toks - needle_b_toks - q_toks - sys_toks - overhead
    if filler_budget < 100:
        filler_budget = 100

    filler_tokens = tokenizer.encode(filler, add_special_tokens=False)[:filler_budget]

    third = len(filler_tokens) // 3
    mid_start = third
    mid_end = 2 * third
    mid = (mid_start + mid_end) // 2

    p1 = tokenizer.decode(filler_tokens[:mid_start])
    p2a = tokenizer.decode(filler_tokens[mid_start:mid])
    p2b = tokenizer.decode(filler_tokens[mid:mid_end])
    p3 = tokenizer.decode(filler_tokens[mid_end:])

    context = p1 + "\n" + needle_a + "\n" + p2a + "\n" + needle_b + "\n" + p2b + p3 + question_str

    messages = [
        {"role": "system", "content": SYSTEM_REASONING},
        {"role": "user", "content": context},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def run_reasoning(llm, tokenizer, cfg, model_key, needles, filler):
    params = SamplingParams(temperature=0, top_p=1.0, max_tokens=150)
    pairs = load_reasoning_pairs()
    results = []

    for ratio in RATIOS:
        target = int(ratio * cfg["ctx"])
        print(f"\n--- reasoning | {model_key} | ratio={ratio} | target={target} ---")

        prompts, meta = [], []
        for pi, pair in enumerate(pairs):
            na = needles[pair["a_idx"]]
            nb = needles[pair["b_idx"]]
            p = build_reasoning_prompt(tokenizer, filler, na["needle"], nb["needle"], pair["question"], target)
            prompts.append(p)
            meta.append({
                "pair_idx": pi,
                "ratio": ratio,
                "question": pair["question"],
                "answer": pair["answer"],
                "needle_a": na["needle"],
                "needle_b": nb["needle"],
            })

        outputs = llm.generate(prompts, params)
        correct = 0
        for m, o in zip(meta, outputs):
            resp = o.outputs[0].text.strip()
            m["response"] = resp
            m["model"] = model_key
            m["task"] = "reasoning"
            m["correct"] = 1 if m["answer"].lower() in resp.lower() else 0
            correct += m["correct"]
            results.append(m)

        print(f"  accuracy: {correct}/{len(meta)} = {correct/len(meta):.3f}")

    return results

def run(model_key, task):
    cfg = MODELS[model_key]
    print(f"loading {cfg['hf']}...")

    tokenizer = AutoTokenizer.from_pretrained(cfg["hf"])
    llm = LLM(
        model=cfg["hf"],
        max_model_len=cfg["ctx"],
        tensor_parallel_size=int(os.environ.get("TP_SIZE", "1")),
        gpu_memory_utilization=0.95,
        trust_remote_code=True,
    )

    needles = load_needles()
    filler = load_filler()

    if task == "single":
        results = run_single(llm, tokenizer, cfg, model_key, needles, filler)
    elif task == "multi":
        results = run_multi(llm, tokenizer, cfg, model_key, needles, filler)
    elif task == "reasoning":
        results = run_reasoning(llm, tokenizer, cfg, model_key, needles, filler)
    else:
        raise ValueError(f"unknown task: {task}")

    out = f"results_{model_key}_{task}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {len(results)} results to {out}")

def run_all(model_key):
    cfg = MODELS[model_key]
    print(f"loading {cfg['hf']}...")

    tokenizer = AutoTokenizer.from_pretrained(cfg["hf"])
    llm = LLM(
        model=cfg["hf"],
        max_model_len=cfg["ctx"],
        tensor_parallel_size=int(os.environ.get("TP_SIZE", "1")),
        gpu_memory_utilization=0.95,
        trust_remote_code=True,
    )

    needles = load_needles()
    filler = load_filler()

    for task in ["single", "multi", "reasoning"]:
        print(f"\n{'='*60}")
        print(f"  RUNNING {task.upper()} for {model_key}")
        print(f"{'='*60}")

        if task == "single":
            results = run_single(llm, tokenizer, cfg, model_key, needles, filler)
        elif task == "multi":
            results = run_multi(llm, tokenizer, cfg, model_key, needles, filler)
        elif task == "reasoning":
            results = run_reasoning(llm, tokenizer, cfg, model_key, needles, filler)

        out = f"results_{model_key}_{task}.json"
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"saved {len(results)} results to {out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODELS.keys()))
    p.add_argument("--task", default="all", choices=["single", "multi", "reasoning", "all"])
    args = p.parse_args()
    if args.task == "all":
        run_all(args.model)
    else:
        run(args.model, args.task)
