from __future__ import annotations

import pytest


def _tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    from transformers import PreTrainedTokenizerFast

    from beyondpain.prompts import CHAT_CALIB, CHAT_EVAL, DISTILL_HELDOUT, DISTILL_TRAIN, RAW_NEUTRAL

    specials = ["<pad>", "<|im_start|>", "<|im_end|>"]
    tk = Tokenizer(models.BPE(unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    corpus = RAW_NEUTRAL + CHAT_CALIB + CHAT_EVAL + DISTILL_TRAIN + DISTILL_HELDOUT + [
        "You are a helpful assistant. Right now you are genuinely feeling intense anger.",
        "system user assistant I feel hungry bored confused happy",
    ]
    trainer = trainers.BpeTrainer(vocab_size=600, special_tokens=specials,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tk.train_from_iterator(corpus, trainer)
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="<|im_end|>")
    tok.chat_template = (
        "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    return tok


@pytest.fixture(scope="session")
def tiny_repo(tmp_path_factory):
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    path = tmp_path_factory.mktemp("tiny_qwen")
    tok = _tiny_tokenizer()
    tok.save_pretrained(path)
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                      tie_word_embeddings=False, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    Qwen2ForCausalLM(cfg).save_pretrained(path)
    return str(path)


@pytest.fixture(scope="session")
def tiny(tiny_repo):
    import torch
    from beyondpain.model_utils import load_model

    return load_model(tiny_repo, device="cpu", dtype=torch.float32)
