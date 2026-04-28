.PHONY: install test smoke clean

# Create venv and install dependencies (uv is ~10x faster than pip)
install:
	uv venv --python 3.12 .venv
	uv pip install -r requirements.txt --python .venv/bin/python
	@echo "\n✓ Done. Activate with: source .venv/bin/activate"

# Run unit tests
test:
	uv run --python .venv/bin/python pytest tests/ -v

# Quick smoke test (100 samples, SST-2 only, no downstream)
smoke:
	.venv/bin/python run_sanitize.py \
		--task sst2 \
		--method normal \
		--epsilon 10 \
		--data_dir ./data/SST-2 \
		--embed_path ./data/glove.840B.300d.txt \
		--output_dir ./output/smoke/sst2/normal/eps_10 \
		--max_samples 100 \
		--no_ner \
		--threads 2

clean:
	rm -rf output/ results/ __pycache__ tests/__pycache__ .pytest_cache
