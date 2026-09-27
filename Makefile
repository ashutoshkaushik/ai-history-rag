# Common project commands. Run `make <target>`.
.PHONY: download ingest ask eval viz

download:        ## fetch every document in corpus/manifest.yaml
	uv run python scripts/download_corpus.py

ingest:          ## build/update the Chroma index (incremental)
	uv run python -m rag.ingest

ask:             ## make ask Q="your question"
	uv run python -m rag.ask "$(Q)" --compare

eval:            ## make eval LABEL=v1-vector [STRATEGY=vector]
	uv run python -m rag.evaluate --label $(or $(LABEL),run) --strategy $(or $(STRATEGY),vector)

viz:             ## rebuild the chunk and embedding viewers
	uv run python scripts/visualize_chunks.py
	uv run python scripts/visualize_embeddings.py
