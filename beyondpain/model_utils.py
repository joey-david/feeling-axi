from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import torch


def load_model(repo: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    kwargs = dict(dtype=dtype, low_cpu_mem_usage=True)
    if device == "cuda":
        kwargs["device_map"] = "auto" if torch.cuda.device_count() > 1 else "cuda"
    try:
        model = AutoModelForCausalLM.from_pretrained(repo, **kwargs)
    except TypeError:  # transformers < 4.56 names the argument torch_dtype
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(repo, **kwargs)
    if device != "cuda":
        model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def get_layers(model):
    inner = model.model
    if hasattr(inner, "language_model"):
        inner = inner.language_model
    return inner.layers


def steering_layer(n_layers: int, frac: float = 0.6) -> int:
    """The upstream steering depth: int(0.6 * n_layers), layer 38 of 64 for Qwen2.5-32B."""
    return max(0, min(n_layers - 1, int(n_layers * frac)))


def input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


@dataclass
class Steer:
    """Adds ``coeff * vector`` to the residual stream output of one decoder layer.

    ``coeff`` is a float or a per-row tensor. ``mask`` (batch x seq, float) limits the
    addition to chosen positions during a full forward; decode steps are always steered.
    The vector may require grad, which is how distill.py trains it.
    """

    model: object
    layer: int
    vector: torch.Tensor
    coeff: float | torch.Tensor = 1.0
    mask: torch.Tensor | None = None
    _handle: object = field(default=None, repr=False)

    def _hook(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        c = self.coeff
        if not torch.is_tensor(c):
            if c == 0:
                return output
            c = torch.tensor(float(c), device=hs.device)
        vec = self.vector.to(hs.device, hs.dtype)
        c = c.to(hs.device, hs.dtype)
        add = c.reshape(-1, 1, 1) * vec.reshape(1, 1, -1) if c.dim() else c * vec.reshape(1, 1, -1)
        if self.mask is not None and hs.shape[1] == self.mask.shape[1]:
            add = add * self.mask.to(hs.device, hs.dtype)[:, :, None]
        hs = hs + add
        return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs

    def __enter__(self):
        self._handle = get_layers(self.model)[self.layer].register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


@contextlib.contextmanager
def maybe_steer(model, layer, vector, coeff):
    if vector is None or (not torch.is_tensor(coeff) and coeff == 0):
        yield None
    else:
        with Steer(model, layer, vector, coeff) as s:
            yield s


def _fold_roles(messages, allow_system: bool):
    if allow_system or not messages or messages[0]["role"] != "system":
        return messages
    rest = [dict(m) for m in messages[1:]]
    rest[0]["content"] = messages[0]["content"] + "\n\n" + rest[0]["content"]
    return rest


def chat_ids(tok, messages, add_generation_prompt: bool = True) -> list[int]:
    """Chat-template token ids; folds the system turn into the first user turn when the
    template rejects a system role (Gemma 2)."""
    for allow_system in (True, False):
        try:
            text = tok.apply_chat_template(
                _fold_roles(messages, allow_system), add_generation_prompt=add_generation_prompt, tokenize=False
            )
            return tok(text, add_special_tokens=False).input_ids
        except Exception:
            if not allow_system:
                raise
    raise RuntimeError("unreachable")


def pad_left(seqs: list[list[int]], pad_id: int, device):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), int(pad_id), dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, L - len(s):] = torch.tensor(s, dtype=torch.long)
        att[i, L - len(s):] = 1
    return ids.to(device), att.to(device)


@torch.no_grad()
def generate(model, tok, prompt_ids: list[list[int]], max_new_tokens: int, batch_size: int = 16,
             do_sample: bool = False, temperature: float = 1.0, top_p: float = 1.0, seed: int = 0) -> list[list[int]]:
    """Batched left-padded generation; returns only the new token ids per row."""
    out: list[list[int]] = []
    dev = input_device(model)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for i in range(0, len(prompt_ids), batch_size):
        chunk = prompt_ids[i:i + batch_size]
        ids, att = pad_left(chunk, pad, dev)
        if do_sample:
            torch.manual_seed(seed + i)
        gen = model.generate(
            input_ids=ids, attention_mask=att, max_new_tokens=max_new_tokens, do_sample=do_sample,
            temperature=temperature if do_sample else None, top_p=top_p if do_sample else None,
            top_k=None, pad_token_id=pad,
        )
        for row in gen[:, ids.shape[1]:].tolist():
            eos = {tok.eos_token_id, pad}
            cut = next((j for j, t in enumerate(row) if t in eos), len(row))
            out.append(row[:cut])
    return out
