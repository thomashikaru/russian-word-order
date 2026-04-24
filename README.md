# Conditional Word Co-occurrence (Russian Sentences)

This project computes conditional word probabilities from a large sentence corpus and uses them to score sentences via a chain-rule style decomposition.

## Mathematical Overview

The corpus is treated at the **sentence level** (set membership), not token-position counts.

For words `a`, `b`, `c`:

- `count(b)` = number of sentences containing `b`
- `count(a,b)` = number of sentences containing both `a` and `b`
- `count(a,b,c)` = number of sentences containing `a`, `b`, and `c`

We compute:

- `P(a | b) = count(a,b) / count(b)`
- `P(a | b,c) = count(a,b,c) / count(b,c)`

Interpretation:

- High `P(a|b)` means: among sentences that contain `b`, `a` often appears too.
- High `P(a|b,c)` means: given that a sentence already contains both `b` and `c`, `a` is likely to co-occur.

For sentence scoring, we use a short-context chain:

- First word: `P(w1)` (estimated as `count(w1) / N`, where `N` is total sentence count)
- Second word: `P(w2 | w1)`
- Third and later words: `P(wi | w(i-2), w(i-1))`

Each step also has surprisal:

- `surprisal = -log2(P)` bits

If a probability is `0`, surprisal is `Infinity`.

## Data and Build

Input dataset (already present):

- `data/russian_sentences.txt`

The build creates a persisted SQLite artifact with:

- unigram sentence counts
- inverted index postings (Roaring bitmaps)

### 1) Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pyroaring
```

### 2) Build the database (no-pairs mode)

```bash
python conditional_cooccurrence.py build-no-pairs \
  --sentences data/russian_sentences.txt \
  --db artifacts/cooccurrence_no_pairs.sqlite \
  --overwrite
```

Notes:

- Omit `--overwrite` if you want to keep an existing DB and write to a new path instead.
- You can add `--max-sentences <N>` for a quick smaller test build.

## Run Sentence-Chain Queries

Prepare an input JSON file with sentence queries, e.g. `data/sentence_test_input.json`:

```json
{
  "db_path": "artifacts/cooccurrence_no_pairs.sqlite",
  "queries": [
    { "id": "s1", "type": "sentence", "sentence": "Она любила его" },
    { "id": "s2", "type": "sentence", "sentence": "Она его любила" }
  ]
}
```

Run:

```bash
python run_query_tests.py \
  --input data/sentence_test_input.json \
  --mode sentence-chain
```

Output is JSON. For each sentence item, fields include:

- `id`
- `type`
- `sentence`
- `probabilities` (chain components in order)
- `surprisals` (per-component `-log2(p)`)
- `surprisal_sum`
- `surprisal_variance`
- `status`
