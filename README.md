# FINAL_CCBD

AI-powered venture due diligence for technical and IP risk analysis using multi-hop reasoning over pitch decks, code repositories, patents, and licensing signals.

This project combines a startup's claims, repository evidence, patent corpus, and dependency metadata into a fused knowledge graph to answer critical due-diligence questions and flag hidden risk before investment.

## What the pipeline does

- Parses startup pitch or whitepaper content
- Indexes repository structure, dependencies, and code semantics
- Extracts patent and legal signals from full-text documents
- Fuses multiple evidence sources into a unified graph
- Resolves entities and traces multi-hop reason paths
- Detects contradictions and risk signals
- Generates explainable due-diligence questions
- Produces an audit trail for transparency and evaluation

## Project structure

```text
.
├── pipeline.py
├── Dockerfile
├── Dockerfile.ollama
├── docker-compose.yml
├── requirements.txt
├── pytest.ini
├── README.md
├── data/
│   ├── raw/
│   └── processed/
├── eval/
├── scripts/
├── src/
├── tests/
├── web/
└── .gitignore
```

## Quick start

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the pipeline in dry-run mode:

```bash
python pipeline.py \
  --pitch data/raw/pitch_decks/halcyon_ai_pitch_deck.pdf \
  --repo data/raw/repositories/open-webui \
  --patents data/raw/patents \
  --dry-run
```

3. Run with an Ollama model:

```bash
ollama pull phi3
python pipeline.py \
  --pitch data/raw/pitch_decks/halcyon_ai_pitch_deck.pdf \
  --repo data/raw/repositories/open-webui \
  --patents data/raw/patents \
  --provider ollama \
  --model phi3
```

4. Run the test suite:

```bash
pytest tests/ -v
```

## Example commands

```bash
# Resume from a later stage
python pipeline.py --start-stage 8

# Stop after graph fusion
python pipeline.py --end-stage 5

# Limit generated questions
python pipeline.py --max-questions 10

# Evaluate generated outputs
python eval/eval_runner.py \
  --questions data/processed/questions.json \
  --ground-truth eval/ground_truth.json
```

## Key outputs

Generated artifacts are stored in `data/processed/`, including:

- `fused_knowledge_graph.json`
- `fused_knowledge_graph.graphml`
- `structured_evidence.json`
- `contradiction_evidence.json`
- `vc_risk_report.json`
- `questions.json`
- `due_diligence_questions.json`
- `audited_vc_report.json`
- `eval_results.json`

## Notes

- The project supports deterministic fallback generation when LLM access is unavailable.
- Local inference can be configured using Ollama or Anthropic API keys.
- The workflow is designed for explainable validation of technical claims, dependency risk, and patent/IP exposure.

## License

This project is intended for research and technical due-diligence workflows. Please review the repository's licensing and data usage constraints before commercial deployment.
