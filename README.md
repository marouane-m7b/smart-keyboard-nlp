# Smart Keyboard · NLP Lab 02

Original instructor-provided Streamlit interface: enter a sentence prefix and view three next-word suggestions with their probabilities. Uses a trigram model with add-k smoothing and hard back-off.

## Run locally

Requires Python 3.12+ and uv.

```sh
uv sync
uv run python -m src.evaluate
uv run streamlit run app.py
```

uv manages dependencies using pyproject.toml and uv.lock. Uvicorn is included as requested, but Streamlit uses its own server.

## Evaluation

The original fixed document split uses 960 training sentences and 240 held-out test sentences. The evaluation trains only on training documents. The original interface trains on the full corpus, as provided by the instructor. Metrics include EOS as a prediction target. Evaluation generates results/metrics.json, results/failures.txt, and results/ngram_results.png.

The synthetic corpus (seed 42) is far more regular than real English, making perplexities low. Its vocabulary has animal, food, person, place, and object groups. Gold POS tags are included; watch is ambiguous between noun and verb. Results illustrate model behavior, not real-world typing performance.

## Deployment

Deploy this repository on Streamlit Community Cloud, selecting app.py and Python 3.12. Dependencies are locked in uv.lock. No API keys or model downloads are required.
