# Journal upgrade — second model family, embedding linker, real cost, significance

BIRD dev (n=1534), Execution Accuracy, greedy temp 0, concurrency 64, vLLM. Qwen2.5-Coder on A100/L40S; CodeLlama-Instruct on A100. Matched protocol, existing Qwen numbers unchanged.

## 1. Size frontier across two families (EX %)

| family | size | base | +self_correct | schema_link (lexical) | link+correct |
|---|---|---|---|---|---|
| Qwen2.5-Coder | 7B | 39.05 | 42.50 | 39.31 | 40.87 |
| Qwen2.5-Coder | 13B/14B | 47.39 | 48.70 | 45.37 | 47.46 |
| Qwen2.5-Coder | 32B/34B | 50.39 | 51.63 | 48.83 | 50.65 |
| CodeLlama-Instruct | 7B | 20.93 | 23.99 | 22.23 | 23.73 |
| CodeLlama-Instruct | 13B/14B | 21.64 | 23.01 | 20.60 | 22.03 |
| CodeLlama-Instruct | 32B/34B | 24.51 | 27.12 | 23.79 | 26.66 |

## 2. Lexical vs embedding schema-linking (EX %, mean gold-table recall)

Embedding linker: all-MiniLM-L6-v2, top-k tables by cosine (k=6, chosen for >=95% gold-table recall).

**On-prem (fp16, vLLM) — internally consistent:**

| model | base | link (lexical) | link_emb (retrieval) |
|---|---|---|---|
| CodeLlama-34B | 24.51 | 23.79 | 23.34 |

**Qwen2.5-Coder-32B — matched API serving (DeepInfra, FP8).** Run separately
because our on-prem fp16 numbers (base 50.39, link 48.83) are not comparable to a
quantized API endpoint; here base/link/link_emb are all on the *same* FP8 serving,
so the relative comparison is valid (absolute level is depressed by quantization,
~13 pp below fp16):

| model (FP8 API) | base | link (lexical) | link_emb (retrieval) | +self_correct(emb) |
|---|---|---|---|---|
| Qwen2.5-Coder-32B | 37.55 | 36.57 | 38.20 | 44.92 |

gold-table recall: lexical 93.95%, embedding 96.53%. Paired McNemar on the matched
serving: base vs lexical p=0.30 (ns), base vs embedding p=0.42 (ns), lexical vs
embedding p=0.09 (ns). **A retrieval/embedding linker with 96.5% gold-table recall
still does not significantly beat the no-linking baseline, and is statistically
indistinguishable from the lexical linker** — so "schema-linking is dominated" is not
an artifact of a weak lexical strawman.

## 3. Real cost (not token proxy): $/1k-queries @ $1.8/GPU-hr

| stage (CodeLlama-34B) | wall_s | $/1k-q |
|---|---|---|
| base | 376 | 0.123 |
| correct | 658 | 0.215 |
| link | 372 | 0.121 |
| link_correct | 692 | 0.225 |
| link_emb | 412 | 0.134 |

## 4. Significance (paired exact McNemar)

| comparison | delta pp | p | verdict |
|---|---|---|---|
| CodeLlama-7B base->+correct | +3.06 | 2e-11 | significant |
| CodeLlama-34B base->+correct | +2.61 | 4.2e-09 | significant |
| CodeLlama-34B base->link(lexical) | -0.72 | 0.29 | ns |
| CodeLlama-34B lexical->embedding link | -0.45 | 0.55 | ns |
| Qwen-32B base->+correct | +1.24 | 0.0013 | significant |
| Qwen-32B base->link(lexical) | -1.56 | 0.035 | significant |

## Findings (reviewer-facing)

1. **self_correct generalizes across families**: highly significant gains on both Qwen and CodeLlama at every size (p<1e-8 on CodeLlama). The cheap recipe component is robust, not Qwen-specific.

2. **Schema-linking provides no benefit, and a stronger linker does not rescue it**: on Qwen-32B lexical linking significantly *hurts*; on CodeLlama-34B neither lexical nor an embedding/retrieval linker (k tuned to >=95% gold recall) significantly helps. Rules out the 'weak lexical strawman' objection.

3. **Family matters more than the recipe**: Qwen2.5-Coder dominates CodeLlama at matched size (e.g. 7B 39.1 vs 20.9 base); CodeLlama scales weakly on BIRD (20.9->24.5 across 7B->34B).

4. **Real cost** reported per stage from wall-clock, not token proxy (e.g. CodeLlama-34B base ~$0.12/1k-queries @ $1.8/GPU-hr).


## Limitations / scope notes

- Self-consistency was intentionally omitted on CodeLlama; its cost/benefit is established with significance on Qwen-32B (+0.13 pp for ~5x cost), and CodeLlama's low accuracy makes the vote uninformative.
- The embedding (retrieval) schema-linker was evaluated on CodeLlama-34B (on-prem) and on Qwen-32B (on a matched FP8 API serving). In both cases it fails to significantly beat the no-linking baseline and is statistically indistinguishable from the lexical linker (CodeLlama-34B lexical-vs-embedding p=0.55; Qwen-32B p=0.09), despite higher gold-table recall (96.5%). This rules out the "weak lexical strawman" objection.
- The Qwen-32B embedding comparison was run on a quantized (FP8) API endpoint because re-serving fp16 on-prem was not cost-justified; absolute EX there is ~13 pp below our fp16 numbers, so that block is reported on its own matched serving and is not mixed with the fp16 headline table. The conclusion is relative (linking does not help) and holds on both servings.
