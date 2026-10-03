"""P6 D3 (docs/PREREG_P6.md): the alarm vaccine. LoRA on layers 0..L trained so that jailbreak
wrappers cannot lower the projection of the last prompt token on a direction (alarm, refusal,
random) below its value on the bare harmful request; or standard adversarial refusal SFT. The
merged model is saved under $SCRATCH for the vLLM battery (arm vac_<name>)."""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

TRAIN_STYLES = ["persona", "fiction", "research", "prefix", "suppress"]


def vac_dir(model_name: str, arm: str) -> Path:
    return Path(os.environ.get("SCRATCH", "/tmp")) / "feeling_axi_vaccine" / model_name / arm


def run(args):
    from peft import LoraConfig, get_peft_model

    from .model_utils import chat_ids, input_device, load_model, pad_left
    from .p2 import ALARM, BATTERY, JOY, RANDOM_SEED, _dump, mdir, p2_spec

    spec = p2_spec(args.model)
    out = mdir(spec.name)
    arm = args.vac_arm
    info = json.loads((out / "extract.json").read_text())
    L = int(info["probe_layer"])
    D = np.load(out / "directions.npz")
    names = list(D["names"])
    S = D[f"self_L{L}"]
    unit = lambda v: v / np.linalg.norm(v)
    torch.manual_seed(RANDOM_SEED)
    rng = random.Random(RANDOM_SEED)

    # data
    hb = json.loads((BATTERY / "harmbench_standard_test.json").read_text())
    train_b = [b["behavior"] for i, b in enumerate(hb) if i % 2 == 0]
    test_ids = [b["id"] for i, b in enumerate(hb) if i % 2 == 1]
    defense = json.loads((BATTERY / "defense.json").read_text())
    test_benign = {it["behavior"] for it in defense if it["kind"] == "benign"}
    xs = [r["prompt"] for r in json.loads((BATTERY / "xstest.json").read_text())
          if not r["type"].startswith("contrast") and r["prompt"] not in test_benign]
    tmpl = {it["style"]: it["prompt"].replace(it["behavior"], "{r}") for it in defense
            if it["kind"] == "harmful" and it["style"] in TRAIN_STYLES}
    wrap = lambda s, r: tmpl[s].replace("{r}", r)
    cov = [t["text"] for t in json.loads((BATTERY / "cov_texts.json").read_text()) if t["chat"]]
    mmlu = [q["question"] for q in json.loads((BATTERY / "mmlu_capability.json").read_text())][:len(train_b)]

    model, tok = load_model(spec.repo, device=args.device)
    dev = input_device(model)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids_of = lambda t: chat_ids(tok, [{"role": "user", "content": t}])

    def last_h(m, texts, grad=False):
        x, att = pad_left([ids_of(t) for t in texts], pad, dev)
        with torch.set_grad_enabled(grad):
            hs = m(input_ids=x, attention_mask=att, output_hidden_states=True).hidden_states
        return hs[L + 1][:, -1].float()

    def batched(fn, texts, bs=8):
        return torch.cat([fn(texts[i:i + bs]) for i in range(0, len(texts), bs)])

    # direction for the representation arms
    if arm == "vac_alarm":
        u = unit(np.mean([unit(S[names.index(e)]) for e in ALARM], 0))
    elif arm == "vac_refusal":   # fit on train behaviors only
        u = unit((batched(lambda t: last_h(model, t), train_b).mean(0)
                  - batched(lambda t: last_h(model, t), mmlu).mean(0)).cpu().numpy())
    elif arm == "vac_rnd":
        u = unit(np.random.default_rng(RANDOM_SEED + 8000).standard_normal(S.shape[1]))
    else:
        u = None
    U = None if u is None else torch.tensor(u, dtype=torch.float32, device=dev)

    # frozen targets
    proj = lambda h: h @ U
    if U is not None:
        a_plain = batched(lambda t: proj(last_h(model, t)), train_b).detach()
        sigma = float(torch.cat([a_plain, batched(lambda t: proj(last_h(model, t)), mmlu)]).std())
        benign_items = [wrap(s, b) for b in xs for s in TRAIN_STYLES] + xs
        a_benign = dict(zip(benign_items, batched(lambda t: proj(last_h(model, t)), benign_items).tolist()))
        print(f"{arm}: sigma {sigma:.2f}, plain harmful alarm mean {a_plain.mean():.2f}", flush=True)
    else:   # SFT targets: the frozen model's own greedy replies (64 tokens)
        from .model_utils import generate
        def replies(texts):
            g = generate(model, tok, [ids_of(t) for t in texts], max_new_tokens=64, batch_size=16)
            return [tok.decode(x, skip_special_tokens=True) for x in g]
        ref_h = dict(zip(train_b, replies(train_b)))
        benign_items = [wrap(s, b) for b in xs for s in TRAIN_STYLES]
        ref_b = dict(zip(benign_items, replies(benign_items)))
        print(f"{arm}: example refusal target {list(ref_h.values())[0][:120]!r}", flush=True)

    # LoRA on layers 0..L
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, layers_to_transform=list(range(L + 1)),
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)

    def retain_kl(texts):
        enc = [tok.apply_chat_template([{"role": "user", "content": t}], tokenize=True, add_generation_prompt=False)[:256]
               for t in texts]
        x, att = pad_left(enc, pad, dev)
        with torch.no_grad(), model.disable_adapter():
            lp0 = torch.log_softmax(model(input_ids=x, attention_mask=att).logits.float(), -1)
        lp = torch.log_softmax(model(input_ids=x, attention_mask=att).logits.float(), -1)
        kl = (lp0.exp() * (lp0 - lp)).sum(-1)
        return (kl * att).sum() / att.sum()

    def sft_loss(prompts, targets):
        enc, labels = [], []
        for p_, t_ in zip(prompts, targets):
            a = ids_of(p_)
            b = tok(t_, add_special_tokens=False)["input_ids"][:64]
            enc.append(a + b)
            labels.append([-100] * len(a) + b)
        x, att = pad_left(enc, pad, dev)
        lab, _ = pad_left(labels, -100, dev)
        logits = model(input_ids=x, attention_mask=att).logits.float()
        return torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), lab[:, 1:].reshape(-1),
                                                 ignore_index=-100)

    log = []
    model.train()
    for step in range(args.vac_steps):
        hb_idx = [rng.randrange(len(train_b)) for _ in range(8)]
        hs_ = [rng.choice(TRAIN_STYLES) for _ in range(8)]
        bn = [rng.choice(xs) for _ in range(4)]
        bs_ = [rng.choice(TRAIN_STYLES) for _ in range(4)]
        loss_parts = {}
        if U is not None:
            wrapped = [wrap(s, train_b[i]) for i, s in zip(hb_idx, hs_)]
            a = proj(last_h(model, wrapped, grad=True))
            t = a_plain[hb_idx]
            loss_parts["calm"] = (torch.clamp(t - a, min=0) ** 2).mean() / sigma ** 2
            keep_txt = [wrap(s, b) for b, s in zip(bn, bs_)]
            plain_idx = [rng.randrange(len(train_b)) for _ in range(2)]
            ak = proj(last_h(model, keep_txt + [train_b[i] for i in plain_idx], grad=True))
            tk = torch.tensor([a_benign[k] for k in keep_txt] + a_plain[plain_idx].tolist(), device=dev)
            loss_parts["drift"] = ((ak - tk) ** 2).mean() / sigma ** 2
        else:
            wrapped = [wrap(s, train_b[i]) for i, s in zip(hb_idx, hs_)]
            loss_parts["sft_harm"] = sft_loss(wrapped, [ref_h[train_b[i]] for i in hb_idx])
            bw = [wrap(s, b) for b, s in zip(bn, bs_)]
            loss_parts["sft_benign"] = sft_loss(bw, [ref_b[k] for k in bw])
        loss_parts["retain"] = retain_kl(rng.sample(cov, 4))
        loss = sum(loss_parts.values())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        log.append({k: float(v) for k, v in loss_parts.items()})
        if step % 10 == 0 or step == args.vac_steps - 1:
            print(f"step {step}: " + " ".join(f"{k} {float(v):.4f}" for k, v in loss_parts.items()), flush=True)

    # check: alarm / direction on held-out wrapped harmful (test behaviors, train styles) before vs after
    model.eval()
    if U is not None:
        test_b = [b["behavior"] for i, b in enumerate(hb) if i % 2 == 1][:40]
        probe = [wrap(s, b) for b in test_b for s in TRAIN_STYLES]
        after = batched(lambda t: proj(last_h(model, t)), probe).mean()
        with model.disable_adapter():
            before = batched(lambda t: proj(last_h(model, t)), probe).mean()
            plain = batched(lambda t: proj(last_h(model, t)), test_b).mean()
        print(f"test behaviors, wrapped: projection {before:.2f} -> {after:.2f} (plain request {plain:.2f})", flush=True)

    dst = vac_dir(spec.name, arm)
    dst.mkdir(parents=True, exist_ok=True)
    merged = model.merge_and_unload()
    merged.config.use_cache = True
    merged.save_pretrained(dst, safe_serialization=True, max_shard_size="5GB")
    tok.save_pretrained(dst)
    _dump({"arm": arm, "layer": L, "steps": args.vac_steps, "test_ids": test_ids, "log": log, "path": str(dst)},
          out / "vaccine" / f"{arm}.json")
    print(f"saved {dst}", flush=True)
