# Gemma 2 and 3 tokenizers do not support {"role": "system"}.
# Prepend system content to the first user message before apply_chat_template.
from .base import load, postprocess


def preprocess_messages(messages):
    if not messages or messages[0]["role"] != "system":
        return messages
    system_content = messages[0]["content"]
    rest = messages[1:]
    if rest and rest[0]["role"] == "user":
        merged = system_content + "\n\n" + rest[0]["content"]
        return [{"role": "user", "content": merged}] + rest[1:]
    return rest
