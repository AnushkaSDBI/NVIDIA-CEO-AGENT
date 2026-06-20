"""
src/llm.py — In-process LLM backend (no Ollama, no server).

Loads the model directly with HuggingFace transformers, on the GPU if present.
This is the backend to use on the university data lab, where Ollama can't be
installed (the ollama.com download is blocked and there's no root). torch and
transformers are already dependencies of this project (sentence-transformers +
the NLI model pull them in), so nothing extra needs installing on the lab's
"PyTorch LLM Focus" image.

It exposes the same single-prompt callable the intelligence layer expects, so it
is a drop-in for the Ollama path — see _get_llm() in intelligence.py.

    from src.llm import chat, chat_json, get_local_llm
    print(chat("You are a strategy analyst.", "Summarize NVIDIA's main risk."))

The model loads once (lazily) on the first call and stays in memory after that.
To persist the model download across server restarts, set this BEFORE python:
    export HF_HOME=$HOME/.hf_cache
"""

import re
import json

import config as cfg

_tok = None
_model = None
_device = None

_SYSTEM = ("You are a precise strategy analyst for an executive intelligence system. "
           "Follow the requested output format exactly. When asked for JSON, return only JSON.")


def _load():
    """Lazy-load the tokenizer + model once; pick GPU automatically, fall back to CPU."""
    global _tok, _model, _device
    if _model is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if _device == "cuda" else torch.float32
        _tok = AutoTokenizer.from_pretrained(cfg.LLM_MODEL)
        _model = AutoModelForCausalLM.from_pretrained(
            cfg.LLM_MODEL,
            torch_dtype=dtype,
            attn_implementation="sdpa",          # PyTorch-native attention; no flash-attn needed
        ).to(_device)
        _model.eval()
    return _tok, _model, _device


def chat(system, user, temperature=None, max_tokens=None):
    """One chat turn -> plain text reply."""
    import torch
    temperature = cfg.LLM_TEMPERATURE if temperature is None else temperature
    max_tokens = cfg.LLM_MAX_TOKENS if max_tokens is None else max_tokens
    tok, model, device = _load()
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    # Some Qwen models support a "thinking" mode (<think>...</think>); turn it off
    # for direct answers. Older models don't accept the flag, so fall back cleanly.
    try:
        text = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens,
                             do_sample=temperature > 0,
                             temperature=max(temperature, 1e-5), top_p=0.9,
                             pad_token_id=tok.eos_token_id)
    gen = out[0][inputs["input_ids"].shape[1]:]            # keep only the newly generated tokens
    return tok.decode(gen, skip_special_tokens=True).strip()


def chat_json(system, user, **kw):
    """chat() but parses a JSON object out of the reply (robust to fences / extra text)."""
    raw = chat(system, user, **kw).strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)               # grab first {...} block
        if m:
            return json.loads(m.group(0))
        raise


def get_local_llm():
    """Return a single-prompt callable: llm(prompt) -> text. Drop-in for the Ollama
    path, since intelligence.py invokes the model as `llm(prompt)` when callable."""
    def _call(prompt):
        return chat(_SYSTEM, prompt)
    return _call


if __name__ == "__main__":
    print(chat("You are a concise equity analyst.",
               "In two sentences, what is NVIDIA's core business?"))
