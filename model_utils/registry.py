MODEL_REGISTRY = {
    # ── Existing models (backward compat) ─────────────────────────────────────
    "llama3-8b": {
        "hf": "meta-llama/Meta-Llama-3-8B-Instruct",
        "ctx": 8192,
        "family": "base",
        "vllm_kwargs": {"trust_remote_code": True},
    },
    "mistral-7b": {
        "hf": "mistralai/Mistral-7B-Instruct-v0.3",
        "ctx": 32768,
        "family": "base",
        "vllm_kwargs": {},
    },
    "phi3-mini": {
        "hf": "microsoft/Phi-3-mini-128k-instruct",
        "ctx": 131072,
        "family": "base",
        "vllm_kwargs": {"trust_remote_code": True},
    },

    # ── Qwen2.5 context-window comparison pairs ────────────────────────────────
    "qwen25-7b-128k": {
        "hf": "Qwen/Qwen2.5-7B-Instruct",
        "ctx": 131072,
        "family": "qwen",
        "vllm_kwargs": {},
    },
    "qwen25-7b-1m": {
        "hf": "Qwen/Qwen2.5-7B-Instruct-1M",
        "ctx": 1000000,
        "family": "qwen",
        "vllm_kwargs": {
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 131072,
            "enforce_eager": True,
        },
    },
    "qwen25-14b-128k": {
        "hf": "Qwen/Qwen2.5-14B-Instruct",
        "ctx": 131072,
        "family": "qwen",
        "vllm_kwargs": {},
    },
    "qwen25-14b-1m": {
        "hf": "Qwen/Qwen2.5-14B-Instruct-1M",
        "ctx": 1000000,
        "family": "qwen",
        "vllm_kwargs": {
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 131072,
            "enforce_eager": True,
        },
    },

    # ── Gemma context-window comparison pair ──────────────────────────────────
    "gemma2-27b": {
        "hf": "google/gemma-2-27b-it",
        "ctx": 8192,
        "family": "gemma",
        "vllm_kwargs": {},
    },
    "gemma3-27b": {
        "hf": "google/gemma-3-27b-it",
        "ctx": 131072,
        "family": "gemma",
        "vllm_kwargs": {},
    },

    # ── OLMo context-window comparison pair ───────────────────────────────────
    "olmo2-32b": {
        "hf": "allenai/OLMo-2-1124-32B-Instruct",
        "ctx": 4096,
        "family": "olmo",
        "vllm_kwargs": {"trust_remote_code": True},
    },
    "olmo3-32b": {
        "hf": "allenai/OLMo-3.1-32B-Instruct",
        "ctx": 65536,
        "family": "olmo",
        "vllm_kwargs": {"trust_remote_code": True},
    },
}
