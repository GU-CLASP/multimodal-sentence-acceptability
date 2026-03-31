# Predicting Sentence Acceptability Judgments in Multimodal Contexts
---
Data and code for multimodal sentence acceptability judgment.

Hyewon Jang, Nikolai Ilinykh, Sharid Loáiciga, Jey Han Lau, Shalom Lappin, **Predicting Sentence Acceptability Judgments in Multimodal Contexts**, _to appear at CMCL 2026_ [(arxiv)](https://arxiv.org/abs/2602.20918). 

Human participants and vision language models (VLMs) rated acceptability of English sentences on a scale of 1 (very unnatural) - 4 (very natural). The sentences were preceded by a relevant visual context (R), irrelevant visual context (I), and no contexts (N). 

---

## Data

### 1. `sentences.csv`
Human acceptability judgment on 75 original English sentences taken from News, Books, and Wikipedia + 225 backtranslated sentences of them.

### 2. `Images`
GPT-5 generated images describing the 75 English sentences.

### 3. `ModelPredictions`
Sentence acceptability ratings provided by 7 VLMs (InternVL3-1B, InternVL3-8B, Qwen2.5-3B, Qwen2.5-7B, llava-1.5-7b, gpt-4o & gpt-4o-mini) averaged across multiple attempts (seeds) for each sentence.

### 4. `ModelLogits`
Logits extracted for each sentence preceded by relevant, irrelevant, null visual contexts for 5 open-source VLMs - with multiple attempts (seeds) for each sentence.

---
## Code

### 1. `generate_gpt.py`
Code for sentence acceptability ratings by gpt-4o & gpt-40-mini.

### 2. `generate_open_source_models.py`
Code for sentence acceptability ratings by InternVL3-1B, InternVL3-8B, Qwen2.5-3B, Qwen2.5-7B & llava-1.5-7b.

### 3. `get_logits_open_source_models.py`
Code for logit extractions from open-source models for each sentence following relevant, irrelevant, and null visual contexts.

### 4. `correlations.ipynb`
Spearman correlations between [human ratings ~ model ratings], [human ratings ~ normalized model logprobs], [model ratings ~ normalized model logprobs].

### 5. `regression.ipynb`
Total least square regressions between ratings in each condition pair ([_N-R_], [_N-I_], [_R-I_]).
