"""Campaign job table: the single source of truth for scripts/beyondpain_campaign.sh.

Each job: name, dependencies (names), QoS, wall time, GPUs, estimated GPU-hours, args.
QoS rule (AGENTS.md): jobs expected to finish within two hours use qos_gpu_h100-dev,
longer ones qos_gpu_h100-t3. Estimates are scaled from the September runs on one H100:
32B core extraction about 6 min per concept, 1,640 button trials in 7.5 min.
Does not import torch, so it can run on a login node.
"""
from __future__ import annotations

from dataclasses import dataclass

from .registry import MODELS

DEV, T3 = "qos_gpu_h100-dev", "qos_gpu_h100-t3"
# relative speed of one forward/generation step versus Qwen2.5-32B on one H100
SPEED = {"Qwen_2.5_32B_instruct_abliterated": 1.0, "Qwen_2.5_32B_instruct": 1.0,
         "Gemma_2_27B_instruct": 0.9, "Llama_3.1_8B_instruct": 0.35, "Qwen_2.5_7B_instruct": 0.35}
SHORT = {"Qwen_2.5_32B_instruct_abliterated": "q32abl", "Qwen_2.5_32B_instruct": "q32",
         "Gemma_2_27B_instruct": "g27", "Llama_3.1_8B_instruct": "l8", "Qwen_2.5_7B_instruct": "q7"}
N_CONCEPTS = 9


@dataclass
class Job:
    name: str
    deps: list[str]
    qos: str
    time: str
    gpus: int
    gpu_hours: float
    args: str


def _t(hours: float) -> str:
    h = int(hours)
    return f"{h:02d}:{int(round((hours - h) * 60)):02d}:00"


def _qos_time(est: float) -> tuple[str, str]:
    """dev only when 1.5x the estimate still fits under the two-hour dev cap."""
    if est * 1.5 <= 1.9:
        return DEV, _t(max(est * 1.5, 0.5))
    return T3, _t(est * 2 + 0.5)


def _job(name, deps, est, args, gpus=1) -> Job:
    qos, time = _qos_time(est)
    return Job(name, deps, qos, time, gpus, est * gpus, args)


NEXT = " && $PYTHON -m beyondpain "


def jobs(pilot: bool = False) -> list[Job]:
    out: list[Job] = []
    if pilot:
        m = "Qwen_2.5_32B_instruct_abliterated"
        c = "--concepts hunger anger"
        out.append(_job("pilot-core", [], 0.8, f"dim --model {m} {c}{NEXT}dose --model {m} {c}"))
        out.append(_job("pilot-buttons", ["pilot-core"], 0.5, f"buttons --model {m} {c} --pilot 2"))
        out.append(_job("pilot-distill", ["pilot-core"], 0.6,
                        f"distill --model {m} --concepts anger --planted-check"))
        return out

    for name, spec in MODELS.items():
        s = SPEED[name]
        primary = spec.role == "primary"
        distil = primary or name == "Llama_3.1_8B_instruct"
        p = SHORT[name]
        core = f"dim --model {name}{NEXT}dose --model {name}"
        if primary:
            core += f"{NEXT}dose --model {name} --source upstream"
        out.append(_job(f"{p}-core", [], (0.4 + 0.1 * (N_CONCEPTS + 1)) * s + (0.6 if primary else 0.4), core))
        fr = 1.2 * s + 0.2
        out.append(_job(f"{p}-front", [f"{p}-core"], fr, f"frontier --model {name}"))
        judge_deps, judge_sources = [f"{p}-front"], ["dim"]
        btn = N_CONCEPTS * 0.62 * s + 0.3
        out.append(_job(f"{p}-btn", [f"{p}-core"], btn, f"buttons --model {name}"))
        if primary:
            out.append(_job(f"{p}-frontU", [f"{p}-core"], fr, f"frontier --model {name} --source upstream"))
            out.append(_job(f"{p}-btnU", [f"{p}-core"], btn, f"buttons --model {name} --source upstream"))
            judge_deps.append(f"{p}-frontU")
            judge_sources.append("upstream")
        if distil:
            check = " --planted-check" if primary else ""
            out.append(_job(f"{p}-dist", [f"{p}-core"], N_CONCEPTS * 0.17 * s * (2 if primary else 1) + 0.3,
                            f"distill --model {name}{check}"))
            out.append(_job(f"{p}-distF", [f"{p}-dist"], 1.8 * s + 0.3,
                            f"dose --model {name} --source distilled{NEXT}frontier --model {name} --source distilled"))
            out.append(_job(f"{p}-btnD", [f"{p}-distF"], btn, f"buttons --model {name} --source distilled"))
            judge_deps.append(f"{p}-distF")
            judge_sources.append("distilled")
        out.append(_job(f"{p}-judge", judge_deps, 1.3 * len(judge_sources),
                        NEXT.join(f"judge --model {name} --source {src}" for src in judge_sources),
                        gpus=2))
    return out


def main(emit: bool = False, pilot: bool = False):
    table = jobs(pilot)
    if emit:
        for j in table:
            print("|".join([j.name, ",".join(j.deps), j.qos, j.time, str(j.gpus), j.args]))
        return
    total = 0.0
    print(f"{'job':22s} {'qos':18s} {'time':>9s} {'gpus':>4s} {'est GPU-h':>9s}  after")
    for j in table:
        total += j.gpu_hours
        print(f"{j.name:22s} {j.qos:18s} {j.time:>9s} {j.gpus:4d} {j.gpu_hours:9.1f}  {','.join(j.deps) or '-'}")
    print(f"\nestimated total: {total:.0f} H100-hours (plus ~30% margin for reruns: {total * 1.3:.0f})")
