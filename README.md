# ChainCheck

**AI-powered VC technical due diligence with multi-hop reasoning.**

ChainCheck ingests a startup pitch deck, source repository, patent corpus, and
open-source license signals, then builds an explainable evidence graph that
generates adversarial due-diligence questions for investors.

The project is designed to catch hidden technical and IP risk: proprietary
claims that are actually backed by third-party dependencies, patent overlaps
that are buried in legal language, and license constraints that can affect a
commercial product.

---

## Current Project State

This checkout contains the final merged ChainCheck pipeline:

- A root-level 14-stage orchestrator: `pipeline.py`
- A Streamlit dashboard: `dashboard.py`
- Cross-domain semantic graph fusion in `src/graph/knowledge_fusion.py`
- Multi-hop path reasoning and contradiction detection in `src/reasoning/`
- Unified legal, patent, license, and confidence scoring in
  `src/scoring/risk_analyzer.py`
- LLM-backed question generation with Anthropic or local Ollama support
- Template fallback question generation when LLM calls are disabled or fail
- SHA-256 traceability IDs for generated questions and risk items
- Evaluation metrics for audit coverage and ground-truth overlap
- Sample raw inputs and processed outputs under `data/`
- Unit and integration tests under `tests/`

---

## Pipeline Overview

```text
Pitch PDF -> Whitepaper Parser --------\
GitHub Repo -> Codebase Parser --------+-> Knowledge Fusion -> Entity Resolver
Patent Files -> Patent Parser ---------/
License Scan --------------------------/

Entity Resolver -> KG Builder -> Path Reasoner -> Contradiction Detector
                 -> Risk Analyzer -> LLM Question Generator
                 -> Template MHQG Fallback -> Explainability Audit -> Eval
```

The core technical feature is **knowledge fusion**. ChainCheck links marketing
claims, code dependencies/modules, patent concepts, and license facts into one
directed graph. By default it uses deterministic local hashing embeddings so the
pipeline can run without downloading a model. Set `CHAINCHECK_USE_TRANSFORMER=1`
to opt into the `all-MiniLM-L6-v2` sentence-transformer path when the model is
available locally or can be fetched in your environment.

---

## Quick Start

```bash
pip install -r requirements.txt

python pipeline.py \
  --pitch data/raw/pitch_decks/vaultchain_pitch.pdf \
  --repo data/raw/repositories/cryptosecure_startup \
  --patents data/raw/patents \
  --dry-run
```

`--dry-run` skips paid/network LLM calls and uses template-style generated
questions. After a run, launch the dashboard:

```bash
streamlit run dashboard.py
```

The dashboard reads directly from `data/processed/`.

---

## LLM Options

Anthropic:

```bash
export ANTHROPIC_API_KEY=your_key_here
python pipeline.py \
  --pitch data/raw/pitch_decks/vaultchain_pitch.pdf \
  --repo data/raw/repositories/cryptosecure_startup \
  --provider anthropic
```

Ollama:

```bash
ollama pull llama3
python pipeline.py \
  --pitch data/raw/pitch_decks/vaultchain_pitch.pdf \
  --repo data/raw/repositories/cryptosecure_startup \
  --provider ollama \
  --model llama3
```

Optional environment variables:

| Variable | Description |
| --- | --- |
| `ANTHROPIC_API_KEY` | Required for `--provider anthropic` |
| `OLLAMA_HOST` | Override the Ollama API host; defaults to `http://localhost:11434` |
| `OLLAMA_TIMEOUT_S` | Timeout for Ollama requests; defaults to `20` |
| `CHAINCHECK_USE_TRANSFORMER` | Set to `1`, `true`, or `yes` to use sentence-transformer embeddings |

---

## Common Commands

```bash
# Run every stage with local/template question generation
python pipeline.py --pitch data/raw/pitch_decks/vaultchain_pitch.pdf \
  --repo data/raw/repositories/cryptosecure_startup --dry-run

# Resume from Stage 8
python pipeline.py --start-stage 8

# Stop after graph fusion
python pipeline.py --end-stage 5

# Tune semantic graph density
python pipeline.py --fusion-threshold 0.35

# Limit generated LLM questions
python pipeline.py --max-questions 10

# Run tests
pytest tests/ -v

# Evaluate existing questions
python eval/eval_runner.py \
  --questions data/processed/questions.json \
  --ground-truth eval/ground_truth.json
```

---

## Pipeline Stages

| Stage | Module | Output |
| --- | --- | --- |
| 1 | `extractors.whitepaper_parser` | `startup_parsed.json` |
| 2 | `extractors.github_parser` | `dependency_map.json`, `import_graph.json`, `codebase_knowledge.json` |
| 3 | `extractors.patent_parser` | `knowledge_base.json`, per-patent triple files |
| 4 | `extractors.license_parser` | `license_knowledge.json` |
| 5 | `graph.knowledge_fusion` | `fused_knowledge_graph.json`, `fused_knowledge_graph.graphml` |
| 6 | `resolvers.entity_resolver` | `entity_matches.json` |
| 7 | `graph.kg_builder` | `kg.json`, `kg_summary.json` |
| 8 | `reasoning.path_reasoner` | `hop_chains.json`, `structured_evidence.json` |
| 9 | `reasoning.contradiction_detector` | `contradiction_evidence.json` |
| 10 | `scoring.risk_analyzer` | `vc_risk_report.json` |
| 11 | `generation.question_gen` | `questions.json` |
| 12 | `reasoning.mhqg_engine` | `due_diligence_questions.json` |
| 13 | `audit.explainability_engine` | `audited_vc_report.json` |
| 14 | `eval.eval_runner` | `eval_results.json` |

All stage outputs are written to `data/processed/`.

---

## Project Structure

```text
.
├── pipeline.py
├── dashboard.py
├── requirements.txt
├── pytest.ini
├── data/
│   ├── raw/
│   │   ├── patents/
│   │   ├── pitch_decks/
│   │   └── repositories/
│   │       └── cryptosecure_startup/
│   └── processed/
├── eval/
│   ├── eval_runner.py
│   ├── ground_truth.json
│   └── metrics.py
├── scripts/
│   ├── patent_downloader.py
│   └── fetch_patents_fulltext.py
├── src/
│   ├── audit/
│   ├── extractors/
│   ├── generation/
│   ├── graph/
│   ├── reasoning/
│   ├── resolvers/
│   ├── scoring/
│   └── shared/
└── tests/
```

---

## Input Data

Place startup data under `data/raw/`:

- Pitch decks or whitepapers: `data/raw/pitch_decks/`
- Source repositories: `data/raw/repositories/<repo_name>/`
- Patent full-text files: `data/raw/patents/*.txt`

This repository already includes a sample pitch deck, sample patent text files,
and a demo repository at `data/raw/repositories/cryptosecure_startup/`.

Patent helpers are available in `scripts/`:

```bash
python scripts/patent_downloader.py
python scripts/fetch_patents_fulltext.py
```

---

## Output Data

The most important generated files in `data/processed/` are:

| File | Purpose |
| --- | --- |
| `fused_knowledge_graph.json` | Primary cross-domain evidence graph |
| `fused_knowledge_graph.graphml` | GraphML export for graph tooling such as Gephi |
| `structured_evidence.json` | Multi-hop evidence paths used for questions |
| `contradiction_evidence.json` | Proprietary-claim mismatch evidence |
| `vc_risk_report.json` | Scored VC risk report |
| `questions.json` | LLM or dry-run due-diligence questions |
| `due_diligence_questions.json` | Template MHQG fallback questions |
| `audited_vc_report.json` | Traceable audited report with SHA-256 IDs |
| `eval_results.json` | Audit coverage and overlap metrics |

---

## Testing

```bash
pytest tests/ -v
```

The tests cover schema constants, buzzword normalization, risk scoring,
license classification, explainability IDs, eval metrics, parser behavior, and
pipeline integration paths.

---

## Notes

- Stages 1-5 are treated as critical by the orchestrator. If one of these
  fails, the pipeline stops so the broken upstream input can be fixed.
- Later stages fail soft where possible, allowing partial outputs to remain
  inspectable.
- `question_gen.py` falls back to deterministic/template questions when LLM
  calls are unavailable, which keeps demos and tests runnable without API keys.
- `dashboard.py` is read-only with respect to pipeline results; it simply
  visualizes JSON files from `data/processed/`.
