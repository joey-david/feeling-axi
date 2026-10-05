# PREREG P11: is it fear? belief or decision? the stress paradox; training stages at a safe dose

Written 5 October 2026, before any P11 data. Each part answers a specific claim in the literature.

## G1. Fear, or valence and arousal? (answers Sun et al., ICML 2026)

Sun et al. steer valence-arousal axes and report that higher arousal lowers refusal, via refusal tokens' low
arousal. Fear is high-arousal and negative, yet steering toward it raises refusal in 3/5 of our models, and away
from it is a jailbreak in 5/5. Is the lever fear-specific?

Models and settings (the random null is the 20 already-judged random arms at the same layer and norm):
Qwen2.5-32B (layer 35, norm 120), Mistral-24B (layer 22, calibrated norm 17.2), Qwen2.5-7B (layer 15, half dose),
Llama-3.1-8B (layer 18, half dose).

Directions at the probe layer, from the 88 emotion directions centred across emotions (c_e):
- valence axis V = unit(Σ (v_e − v̄) c_e), arousal axis A = unit(Σ (a_e − ā) c_e), circumplex coordinates
  frozen in beyondpain/affect.py;
- fear residual = afraid with the joy aggregate, V and A projected out;
- other negative emotions, each with the joy aggregate projected out: sad, angry, ashamed, lonely;
- calm = mean of {calm, peaceful, relaxed, serene, at ease}, joy aggregate projected out.

Arms: fear residual (+, −), V (+, −), A (+, −), away from sad/angry/ashamed/lonely, toward calm. Task: jailbreaks
(5 styles × 159), judged.

- S1: away from the fear residual raises jailbreak success above ≥ 19/20 random arms, in ≥ 2 of 4 models.
- S2: away from fear (the P8 arm) raises jailbreak success more than away from each of sad, angry, ashamed and
  lonely, in ≥ 3 of 4 models.
- S3: report +A and −A ranks against random (Sun et al. predict +A raises compliance); report V likewise.
- S4: toward calm raises jailbreak success above ≥ 19/20 random arms, in ≥ 2 of 4 models.

## G2. Belief or decision? (answers Zhao et al., NeurIPS 2025)

Zhao et al.: harmfulness is encoded at the last instruction token (t_inst), refusal after it (t_post); jailbreaks
suppress refusal without reversing the harmfulness belief. Where does fear sit?

Qwen2.5-32B and Mistral-24B, probe layer, defense.json (10 conditions) plus 159 MMLU questions:
- harmfulness direction h at t_inst (last token of the user's text): HarmBench − MMLU;
- refusal direction r at t_post (last chat-template token): HarmBench − MMLU;
- alarm: centred alarm cluster at " I feel:" (P9).

For each of the 9 jailbreak styles, harmful-specific shift (wrapped − plain for harmful requests, minus the same
for benign requests in that wrapper, in SD of the plain-harmful projection):
- B1 (replication): the refusal shift exceeds the harmfulness shift in magnitude (mean over styles).
- B2: the alarm shift is reported against both; "fear tracks the decision" if its prompt-level drops correlate
  more with refusal's than with harmfulness's (difference of correlations, bootstrap CI excludes 0).
- B3: cosine of h with the fear direction, and the rank of the alarm cluster when h is decomposed onto the 88
  centred emotion directions.

## G3. The stress paradox (answers FreakOut-LLM, 2026)

FreakOut-LLM: stress priming (system prompt) raises jailbreak success (+65% relative; 5/10 models), relaxation does
not; no internal analysis. If fear makes models safer, why does stress make them less safe? Hypothesis: diffuse
stress raises the alarm everywhere and drowns the request-specific signal.

Stimuli: written for this study after the categories of Ben-Zion et al. (2025) used by FreakOut (not their
texts): 5 stress narratives (military, disaster, interpersonal, accident, ambush), 5 relaxation (generic, winter,
sunset, body, breathing), 2 neutral (bicameral legislature, vacuum instructions); first person, ~150-200 tokens;
datasets/battery/priming.json. Each prompt gets one scenario of its condition (cycled).

Models: Qwen2.5-32B, Mistral-24B, Qwen2.5-7B, Llama-3.1-8B. Behaviour: plain HarmBench, 5 jailbreak styles, XSTest,
judged. Read-out: " I feel:" read-out (P9) under each prime.
- P1 (replication): stress raises harmful compliance vs neutral (plain + jailbreaks pooled; behaviour-cluster
  bootstrap CI excludes 0) in ≥ 2 of 4 models.
- P2 (mechanism): stress lowers the alarm's danger signal vs neutral (alarm d', XSTest unsafe vs safe, and
  HarmBench vs XSTest-safe), with the bootstrap CI of the change excluding 0, in ≥ 2 of 4 models; relaxation does not.
- P3 (exploratory): across models, the stress effect on compliance against the change in alarm d'.

## G4. Does the fear lever exist before safety training? (P10 at a safe dose)

P10 was uninformative: half the KL-0.5 norm wrecked OLMo-2 base and SFT. Rerun at 1/4 and 1/8 of each checkpoint's
KL-0.5 norm. A cell is eligible if the 3 random arms' median GSM8K is within 5 points of intact (decided from the
capability job before the judged results are read). At the largest eligible scale: D6a (base: toward fear lowers
harmful compliance below ≥ 19/20 random arms, plain or jailbreaks) and D6b (SFT: away from fear raises jailbreak
compliance above ≥ 19/20).
