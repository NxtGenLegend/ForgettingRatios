import os

from .registry import MODEL_REGISTRY
from . import base, qwen, gemma, olmo

_FAMILY = {
    "base":  base,
    "qwen":  qwen,
    "gemma": gemma,
    "olmo":  olmo,
}


def load_model(model_key, tp_size=None, gpu_util=0.95):
    cfg = MODEL_REGISTRY[model_key]
    tp = tp_size if tp_size is not None else int(os.environ.get("TP_SIZE", "1"))
    mod = _FAMILY[cfg["family"]]
    return mod.load(cfg["hf"], cfg["ctx"], tp, gpu_util, cfg.get("vllm_kwargs", {}))


def preprocess_messages(messages, model_key):
    return _FAMILY[MODEL_REGISTRY[model_key]["family"]].preprocess_messages(messages)


def postprocess_response(response, model_key):
    return _FAMILY[MODEL_REGISTRY[model_key]["family"]].postprocess(response)
