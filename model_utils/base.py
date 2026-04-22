from transformers import AutoTokenizer
from vllm import LLM


def load(hf_id, ctx, tp_size, gpu_util, extra_vllm_kwargs):
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    llm = LLM(
        model=hf_id,
        max_model_len=ctx,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=gpu_util,
        **extra_vllm_kwargs,
    )
    return tokenizer, llm


def preprocess_messages(messages):
    return messages


def postprocess(response):
    return response
