# Local LLM Chatbot & Data Tasks

Solutions to the five tasks in [`task-description.txt`](task-description.txt). Everything runs locally:
the git repository has **no remote** and the chatbot never sends data off the machine.

| Task | What | Where |
|---|---|---|
| 1 | Local GPU-accelerated LLM chatbot (Gradio + Ollama) | [`gradio_app.py`](gradio_app.py), [`src/chatbot/`](src/chatbot) |
| 2 | Binary classification of `Type` | [`src/classification/`](src/classification), [`reports/task2/`](reports/task2) |
| 3 | 5 alternatives per part from `DESCRIPTION` + chatbot integration | [`src/parts/`](src/parts), [`reports/task3/`](reports/task3) |
| 4 | Hour difference between two timestamps | [`src/timediff/timediff.py`](src/timediff/timediff.py) `hours_between` |
| 5 | Hour difference counting only 09:00–17:00 on weekdays | same file, `business_hours_between` |

## Quick start

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# LLM runtime (macOS: brew install ollama, or the Ollama app)
ollama serve &                       # skip if the Ollama app is running
ollama pull gemma4:e2b-it-qat        # chat model, ~1.7 GB in GPU memory
ollama pull embeddinggemma-2:270m    # only needed for the optional embedding evaluation

python gradio_app.py                 # Task 1 + 3 -> http://127.0.0.1:7860
python -m src.classification.train   # Task 2  (~1 min)
python -m src.parts.eda              # Task 3: descriptive analysis
python -m src.parts.find_alternatives  # Task 3: reports/task3/alternatives.csv
python -m src.parts.evaluate --embeddings  # Task 3: compare similarity variants
pytest                               # 33 tests (tasks 1, 3, 4, 5)
```

## Machine used

| | |
|---|---|
| Computer | Apple MacBook (Apple **M1**), macOS 26.7.1 |
| CPU | 8 cores (4 performance + 4 efficiency) |
| GPU | Apple M1 integrated GPU, **7 cores**, Metal 4 |
| Memory | **8 GB** unified memory (shared by CPU and GPU) |
| Runtime | Ollama 0.40.2 (llama.cpp backend, Metal acceleration), Python 3.11.8 |

`ollama ps` reports the model as **`100% GPU`**: all layers are offloaded to the M1 GPU through Metal.
Measured generation speed is **~29–30 tokens/s**, and a typical tool-using answer arrives in 5–12 s.
The app shows tokens/s for every reply in the settings panel.

---

## Task 1: Chatbot

### Features (focus on ease of use)
- Familiar chat layout with **streaming** answers and clickable **example prompts** on an empty chat.
- **Tool transparency.** When the model looks something up, a collapsible "🔧 Tool used" message in the chat shows the arguments and the raw result. You can see where an answer came from instead of having to trust it.
- Settings panel: model picker (any locally pulled Ollama model), tools on/off, an optional *reasoning mode* (Gemma's thinking, which can also be shown), a temperature slider, an editable system prompt, **New chat**, and **Export** to Markdown.
- Clear error messages for the usual setup problems, e.g. Ollama not running or no model pulled, each with the command that fixes it.
- Speed statistics (tokens, tok/s, seconds) for the last reply.

- Built-in copy buttons; reasoning (when shown) appears as a collapsible "💭 Reasoning" message.
- The app binds to `127.0.0.1` only, with `share=False`, so it is never exposed publicly. Gradio also gives it an automatic API (`gradio_client`), which is useful for scripting and tests.

### Architecture
```
gradio_app.py (Gradio UI)  ──►  src/chatbot/engine.py  ──►  Ollama server (localhost:11434, Metal GPU)
                                      │  tool calls
                                      ▼
                               src/chatbot/tools.py  ──►  src/parts (Task 3), src/timediff (Tasks 4/5)
```
`engine.chat_stream` does not depend on the UI. It streams events (`content`, `thinking`, `tool`, `stats`)
and runs the tool-call loop: when the model asks for a tool, the engine executes it, feeds the result
back and continues, for up to 4 rounds. This keeps the UI thin and lets the engine be unit-tested with a
fake client (`tests/test_engine.py`, `tests/test_gradio_app.py`).

### Model choice: `gemma4:e2b-it-qat` (Google Gemma 4 "E2B", instruction-tuned, quantisation-aware 4-bit)
- **Fits the hardware.** With 8 GB of *shared* memory, the model has to leave room for macOS, the browser and Python. E2B uses about 1.7 GB of GPU memory (4.6 B parameters stored, ~2 B "effective" per token). 7–8 B models (Llama 3.1 8B, Qwen 2.5 7B) need about 5 GB at Q4 and would cause swapping.
- **Native tool calling.** Ollama lists `tools` and `thinking` among its capabilities. In testing it reliably called the right tool with correct arguments, e.g. `find_alternative_parts(part_id="A100", k=3)`. Tool calling is what makes the Task 3 integration possible.
- **QAT quantisation.** The model was trained to be quantised, so the 4-bit version loses far less quality than post-hoc quantisation of a similarly sized model.
- **Licence and speed.** It uses the Apache 2.0 licence and runs at ~30 tok/s on an M1, which feels interactive.
- **Alternatives considered:**
  - *Llama 3.2 3B*: good, but weaker at tool use and it has an older knowledge cutoff.
  - *Qwen 2.5 3B*: comparable.
  - *Phi-3.5 mini*: no native tool calling in Ollama.

  The model is chosen in the settings panel, so you can swap in any other pulled model without changing code.

### Package choices
| Package | Why |
|---|---|
| **Ollama** (github.com/ollama/ollama) + `ollama` Python client (PyPI, official) | The most widely used local LLM runtime. It is built on llama.cpp with first-class Apple Metal support, handles model download, quantised weights and memory management, and its client turns plain Python functions with docstrings into tool schemas. Alternatives: `llama-cpp-python` (more manual setup), Hugging Face `transformers` + PyTorch MPS (much slower and more memory-hungry on 8 GB), MLX (Apple-only, weaker tool-calling support). |
| **Gradio** (PyPI) | Ready-made chat component (`gr.Chatbot`) with streaming and collapsible tool/reasoning messages, pure Python with no front-end build, and an automatic API client for testing. Streamlit was considered (and prototyped) but Gradio's in-chat tool messages and API client fit a tool-using chatbot better. |
| pandas, NumPy, scikit-learn, matplotlib, pytest | Standard, well-maintained scientific Python stack (PyPI). Versions are pinned in `requirements.txt`. |

### Version control
The repository was initialised locally (`git init`, branch `main`, **no remote**). Work was committed in small,
self-contained steps: data → Tasks 4/5 → Task 2 → Task 3 → chat engine → UI → tests → docs (see `git log`).
`.gitignore` excludes the virtualenv, caches and `.DS_Store`.

---

## Task 2: Binary classification of `Type`

Code: `src/classification/data.py` (loading, cleaning, joining) and `train.py` (models). Outputs:
`reports/task2/metrics.json` and `task2_results.png`.

### Data findings and handling
1. **Duplicates.** Each table has 4,070 rows but only 3,700 unique IDs, because **370 rows are exact duplicates**.
   - Joining as-is would be many-to-many (4 rows per duplicated ID).
   - It would also leak identical rows into both the training and test sets.
   - Fix: drop exact duplicates, then join one-to-one on `ID` (checked with `validate="one_to_one"`). Result: **3,700 rows**.
2. **Redundant columns.**
   - `ERZ` is a 1:1 relabelling of `GJAH` (oooo↔www, rrr↔pppp, ii↔nnn, with the same missing values).
   - `LPI` is exactly `NKJUD × 10,000`.
   - Both were dropped, leaving 15 features.
3. **Class imbalance.** `y` makes up 92.5% of rows and `n` 7.5%. A majority-class model therefore scores 92.5% accuracy while being useless, so:
   - `n` is treated as the positive class;
   - models use class weights;
   - the main metric is **PR-AUC**, plus F1 and balanced accuracy for `n`;
   - the decision threshold is tuned on out-of-fold predictions, not on the test set.
4. **Missing values.**
   - `ZIK` is 58% missing, so it gets an explicit `missing` category: missingness itself is informative.
   - Other categoricals get the same `missing` category.
   - Numeric gaps are filled with the median plus a missing-indicator column.
   - `CDx` and `NKJUD` are heavily skewed, so they are log-transformed for the linear model.

### Method
- 80/20 stratified hold-out split. Model selection uses 5-fold stratified CV on the training part only.
- Models compared:
  - majority baseline;
  - logistic regression (one-hot encoding plus scaling);
  - random forest;
  - histogram gradient boosting.

### Results

| Model | CV PR-AUC | Test ROC-AUC | Test PR-AUC | Test F1 (n) | Test balanced acc. |
|---|---|---|---|---|---|
| Majority baseline | 0.074 | 0.500 | 0.074 | 0.000 | 0.500 |
| Logistic regression | 0.629 | 0.951 | 0.677 | 0.688 | 0.879 |
| **Random forest (selected)** | **0.932** | **0.996** | **0.967** | **0.932** | **0.936** |
| Hist. gradient boosting | 0.910 | 0.976 | 0.919 | 0.863 | 0.898 |

**Random forest on the test set (740 rows):**
- It labelled 48 of 55 `n` rows correctly, so recall for `n` is 0.87.
- It produced **0 false alarms** (precision 1.00) and 99.1% accuracy overall.

**What drives the predictions (permutation importance):**
- `VOL` dominates: when `VOL = f` the rate of `n` is 42%, against 1.6% when `t`.
- It is followed by `TRE`, `VBNM`, `UKL` and `NKJUD`.
- The large gap between the linear and tree models shows that the decision boundary is non-linear, built from interactions between features.

**Possible next steps:**
- tune hyperparameters (e.g. Optuna);
- use repeated CV for tighter confidence intervals;
- calibrate the probabilities if they will be used as risk scores.

---

## Task 3: Alternative parts

### Descriptive analysis (`python -m src.parts.eda`, full output in `reports/task3/eda.txt`)
- 998 parts (A1–A998) with 32 columns. Every structured column has 12–89% missing values.
- Descriptions are 43–205 characters long (median 165). They are semi-structured spec strings, e.g.
  `Indicator Red Fast Movement 1.6A 250V Holder Plastic 5 X 20mm Ceramic … Electric Indicator, Very Fast Blow, 1.6A, 250VAC, 1500A (IR), Inline/holder, 5x20mm`.
- Main families: *Indicator Chip* (343, SMD), *Indicator Red* (171, 5×20 mm holder), *Indicator SubRed* (73).

**Findings and difficulties, and how they are handled**

1. **A third of the descriptions are missing.** 335 of 998 (33.6%) have no `DESCRIPTION`, and 6–9 parts have practically no information at all.
   - *Handling:* for parts without a description, a text is built from their structured attributes (current, characteristic, voltage, size, material, mounting, …) so they can still be matched.
   - Every result carries a `matched_on` / `description_source` flag (`description` or `attributes`) so lower-confidence matches are visible.
   - Parts with no information at all are never recommended.
2. **Duplicates and contradictions.**
   - Only 582 of the 663 descriptions are distinct; 126 parts share a description with another ID, which makes those perfect alternatives (score ≈ 1).
   - Each description looks like **two concatenated sources** that sometimes disagree:
     - the current is stated twice in 517 descriptions and conflicts in 13 (e.g. `0.08A … 0.8A`);
     - the blow speed is stated twice in 476 and conflicts in 98 (e.g. *Very Fast Movement … Fast Blow*).
   - *Handling:* the first-stated value (the product-title part) is used for the numeric comparison. In production such records should be flagged for data-quality review.
3. **Inconsistent formatting, and numbers that carry the meaning.**
   - The same thing is written in different ways: `5 X 20mm` vs `5x20mm`, `250 V` vs `250V`, `Slow Blow` vs `Time Lag`, `SMD` vs `Surface Mount`.
   - Units are embedded in values (`125Cel`, `1.6A`).
   - Bag-of-words treats `1.6A` and `2A` as unrelated tokens, yet the rating is what matters most for a replacement part.
   - *Handling:* normalise units and synonyms, use character n-grams, and add an explicit **numeric current-similarity** term.

### Solution (`src/parts/similarity.py`)
1. **Normalise** the text: lower-case it, unify units and dimensions, and map synonyms.
2. **TF-IDF** on word 1–2-grams **plus character 3–5-grams**, then cosine similarity.
   - Word n-grams capture terms like *very fast*, *surface mount* and *5x20mm*.
   - Character n-grams cope with formatting variants and partial codes (`cULus`/`UL`, `0.25a`/`0.2a`).
3. **Current similarity:** the rated current is parsed from the text, ignoring the breaking capacity `(IR)`. It is compared as the ratio `min/max`, so 2 A vs 2 A scores 1, 1 A vs 2 A scores 0.5, and 1 A vs 10 A scores 0.1.
4. **Score** = 0.7 × text + 0.3 × current. The part itself is excluded, and the top 5 are returned with scores.

`reports/task3/alternatives.csv` holds the 5 alternatives for every part (columns: `ID, rank, alternative_ID, score, query_source`).
The same class also answers free-text queries (`search("time lag 2A 250V 5x20mm glass")`).

**Why this approach**
- The descriptions are short, technical and keyword-heavy. Lexical similarity is precise on them, deterministic and explainable (you can see which tokens match).
- It is fast: fitting takes under a second on a laptop, and it needs no GPU or external service.
- Numbers matter for parts, and neither TF-IDF nor embeddings handle them well, so they get their own term.

**Evaluation.** There are no labelled "true alternatives", so a proxy was built from the structured columns (`python -m src.parts.evaluate --embeddings`, output in `reports/task3/evaluation.txt`).
- For the 663 parts with a real description, it measures how often the top-5 alternatives share the same *rated current*, *characteristic*, *mounting* and *size*.
- Those columns are not used by the matcher for these parts, so they are an independent check.

| Variant | Current | Characteristic | Mounting | Size | All attributes match | 
|---|---|---|---|---|---|
| Random | 0.04 | 0.29 | 0.46 | 0.12 | 0.00 |
| TF-IDF word+char only | 0.48 | 0.73 | 0.97 | 0.73 | 0.12 |
| **TF-IDF + current, 0.7/0.3 (default)** | **0.69** | 0.63 | 0.93 | 0.62 | **0.14** |
| embeddinggemma (semantic embeddings) only | 0.30 | 0.82 | 0.97 | 0.79 | 0.09 |
| 0.5 TF-IDF + 0.5 embeddings, + current | 0.76 | 0.59 | 0.91 | 0.58 | 0.15 |

Best achievable "all attributes match" (oracle): **0.23**. For most parts fewer than five genuine
drop-in replacements exist in the catalogue, so the score should be shown alongside each match, as the tool does, rather than presenting five equally good matches.

**Conclusions from the evaluation:**
- Semantic embeddings capture the part *type* well but ignore the electrical rating.
- Adding the current term gives a large gain in rating agreement (0.48 → 0.69).
- Fusing in embeddings adds only a little (0.143 → 0.154 on all attributes) at the cost of a running embedding server, so the default stays TF-IDF + current. The weight is a single parameter (`text_weight`).

**Next steps with more time:**
- hard filters chosen by the user (e.g. "same voltage or higher", "same mounting");
- parse voltage and breaking capacity as well;
- get engineers to label a small set of pairs to tune the weights properly.

### Integration into the chatbot (implemented)
- `src/chatbot/tools.py` exposes `find_alternative_parts(part_id, k)` and `search_parts(description, k)` as **LLM tools**. Tasks 4/5 are exposed the same way as `hours_between` / `business_hours_between`.
- The similarity index is built once and cached (`lru_cache`).
- The model decides when to call a tool. The tool returns JSON (IDs, scores, descriptions, match source), which the model then explains or summarises as a table.
- The deterministic retrieval code computes the IDs and scores, so the LLM never invents part numbers. The user can check the raw result in the "🔧 Tool used" expander.

Example prompts:
- "Suggest 5 alternatives for part A100"
- "I need a time lag 2A 250V 5x20mm glass part"
- "Why is A17 a good replacement for A100?" (multi-turn, using the tool results)

---

## Tasks 4 & 5: Hour differences (`src/timediff/timediff.py`)

```python
>>> hours_between("2022/02/15 00:05", "2022/02/15 01:00")
1
>>> business_hours_between("2022/02/18 16:00", "2022/02/21 10:00")  # Fri 16-17 + Mon 9-10
2
```
- **Input:** timestamps in the format `YYYY/MM/DD HH:MM`. Invalid input raises a clear `ValueError`.
- **Order:** the argument order does not matter; the absolute difference is returned.
- **Rounding:** results are rounded to whole hours, with **half rounded up** (`ROUND_HALF_UP`). Python's `round()` uses banker's rounding, which would turn 2.5 h into 2.
- **Task 5:**
  - Each calendar day in the range is clipped to its 09:00–17:00 window.
  - Saturdays and Sundays are skipped.
  - The overlapping minutes are summed, and only then rounded, so partial hours on different days add up correctly.
  - The working window and an optional `holidays` list are parameters.
- **Tests:** 21 tests in `tests/test_timediff.py` cover the task example, rounding edges, reversed order, overnight spans, weekends, full weeks, leap days and holidays.
