# AI approach: what is used, why, and what it is worth

## 1. Architecture in one picture
```
 question ──► Gemma 4 (reasoning + tool choice) ──► tools ──► evidence (numbers + maps)
                    ▲                                   │
                    └──────── reads results & images ◄──┘
 tools = classical evidence (interpret.py, interp_tools.py)
       + AI recognition   (ai_interp.py: 3D U-Net, k-means)
 data  = RAW (SEG-Y, LAS, deviations, checkshots) + INTERPRETED (fault/polygon shapefiles, horizon)
```
Nothing is trained on the team's data. Gemma is not fine-tuned. The copilot gathers evidence, then Gemma reasons over it.

## 2. The AI components

| Component | What it does | Why this choice |
|---|---|---|
| **Gemma 4 26B-A4B-it** (provided, OpenAI-compatible API) | Understands the question, picks and chains tools, reads returned numbers and map images, writes the step-by-step explanation with confidence and risks | Provided by the organisers; open-weight model, so the same design can run inside a company network and keep data in-house. Mixture-of-experts (~4B active parameters, per the model name) keeps answers fast on a shared server. Native tool calling and image input are what make an agent possible. |
| **Tool-grounded reasoning** (agent.py) | Every number must come from a tool; the system prompt forbids invented values and requires SUPPORTED / CONTRADICTED / NOT AVAILABLE per trap type | LLMs hallucinate numbers. Grounding turns Gemma into an explainer of verifiable evidence rather than a source of facts. |
| **3D U-Net fault detector** (PyTorch, ai_interp.py) | Segments fault planes in the full stack, then counts, orients and locates them | A learned detector is the standard approach in published seismic-AI work. It is trained on synthetic seismic with known faults, so no field labels are needed (there are none). Falls back to a coherence method if weights or PyTorch are missing, and says which ran. The bundled weights are a quick 300-step model; retraining with more steps on a GPU improves it. Quality figures measured so far are on synthetic data only. |
| **k-means seismic facies** | Unsupervised classes from the sand-probability / impedance / Vp/Vs / stack attributes, related to wells | No labels needed; gives a second, independent view of lateral facies change. |
| **Classical evidence engines** (interpret.py, interp_tools.py) | Log-to-time conversion, gas flagging, horizon closure detection, updip pinch-out test, fault-bounded fraction, polygon vs raw sand probability, interpreted vs automatic faults | Transparent, deterministic and auditable. A geoscientist can check the rule. This is the "evidence" half; the AI parts are the "recognition" half. |

### Why not the alternatives
- **Fine-tune Gemma on the data:** there are no labelled examples (no tops, no reports) and 13 wells would overfit; it would also hide the evidence.
- **Pure LLM with no tools:** cannot read SEG-Y or shapefiles and would invent numbers.
- **Large pretrained seismic foundation model:** not available offline here; the U-Net can be swapped for one later without changing the tools or the agent.

## 3. How raw and interpreted data are combined
1. Interpreted data is the geoscientist's judgement: used first for structure (horizon, faults) and facies (channel / overbank polygons).
2. Raw data validates it: sand probability inside vs outside channel polygons; interpreted faults vs AI-detected faults; well logs vs polygon membership and horizon time.
3. Agreement raises confidence, disagreement is reported rather than hidden.
4. New files dropped in either folder are ingested incrementally, so the same questions can be asked again on new data.

Trap logic (assess_traps): four-way closure (closed high, height, spill), fault-dependent closure (share of outline bounded by faults), stratigraphic updip pinch-out (sand-body edge facing updip, next to fine-grained polygons, orientation vs dip). Unconformity and reef are marked NOT AVAILABLE unless the data supports them.

## 4. Business value (and how it maps to the judging criteria)
Stated qualitatively. We have not measured time or cost savings; a pilot on the team's own workflow would be needed to put numbers on them.

| Value | How the design delivers it | Criterion |
|---|---|---|
| **Faster screening** | One question triggers the multi-step workflow a geoscientist does by hand (logs to time, gas zone, seismic tie, map, closure check) | Business Impact, Solves Challenge |
| **Consistent, auditable reasoning** | Each answer lists the tools used, numbers, and what would change the conclusion; rules are readable code | Technical Design, Business Impact |
| **Fewer missed contradictions** | Cross-validation of interpretation against raw data and against automatic detection flags disagreements early | Business Impact |
| **Works on new data without rebuilding** | Incremental ingestion, flexible naming, optional config; raw-only, interpreted-only or both | Execution, Technical Design |
| **Data stays in-house** | Open-weight model and local tools; data is read in place, never copied or sent out | Business Impact |
| **Usable by non-programmers** | Chat UI with data panel, scan button, maps in the conversation | UX |
| **Extensible** | New tool = one function plus one line of parameters; U-Net replaceable | Innovation, Technical Design |

## 5. Limits to state when presenting
- Screening scores are not probabilities of success.
- Time-depth for wells without checkshots is low confidence.
- Fault seal and depth conversion are not assessed.
- Tested end-to-end on synthetic interpreted data; real file attributes may need `data_config.json`.
