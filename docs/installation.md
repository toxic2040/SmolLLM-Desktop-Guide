# Contributing to SmolLLM-Desktop-Guide

Thank you for your interest in contributing to SmolLLM-Desktop-Guide! This repository focuses on the rigorous design and rollout of a local LLM hybrid optimized for Bitcoin technical analysis (TA), emphasizing Retrieval-Augmented Generation (RAG) on personal datasets (~5000 images/charts + 600MB text/price data, such as timestamped OHLCV CSVs) and LoRA fine-tuning for custom TA workflows (e.g., pattern recognition like bullish flags at $66k or generating forecast drafts). For broader appeal, contributions can generalize to stocks/altcoins (e.g., via asset_type flags) or other domains (e.g., image/text knowledge bases).

We welcome enhancements that improve feasibility, such as hardware branches (NVIDIA/CPU), RAG optimizations for chronological TA queries (e.g., filtering by timestamps to avoid biases), or integrations from cross-referenced projects (e.g., llmware for advanced embeddings, LlamaFactory for QLoRA variants). All changes must maintain "smol" principles: low resource use, isolation (virtual envs/Docker), and rigorous testing on synthetic Bitcoin TA examples.

## Code of Conduct

By contributing, you agree to uphold a respectful and inclusive environment. Report unacceptable behavior to the maintainers.

## How to Contribute

### 1. Reporting Issues
- **Search Existing Issues**: Check if your issue exists via GitHub search.
- **Create a New Issue**: Use templates in `.github/ISSUE_TEMPLATE/` (e.g., bug_report.md for ROCm quirks, feature_request.md for TA extensions like real-time Coingecko proxies).
- **Details Required**: 
  - Environment: OS (e.g., Linux Mint 22.3), hardware (e.g., RX 7900 GRE), ROCm version.
  - Reproduction Steps: e.g., "Run fine_tune_lora.py on btc_dataset_sample; OOM at batch=4."
  - Bitcoin TA Context: If relevant, describe impact on RAG (e.g., "Fails to embed timestamped prices") or fine-tuning (e.g., "LoRA adapters ignore chart patterns").
  - Broader Appeal: Suggest generalizations (e.g., "Adapt metadata for stock tickers").
  - Logs/Screenshots: Paste errors (e.g., rocm-smi output), dataset snippets (anonymized).

### 2. Submitting Pull Requests (PRs)
- **Fork and Branch**: Fork the repo, create a feature branch (e.g., `feat/rag-timestamp-filter`).
- **Commit Guidelines**: 
  - Messages: Descriptive, e.g., "feat: Add timestamp filtering to build_rag.py for chronological TA queries."
  - Scope: Small, focused changes (e.g., one script enhancement per PR).
- **Code Style**:
  - Python: PEP8 (use black/flake8); docstrings for all functions (e.g., in prep_data.py, detail JSON schema for image-analysis-metadata pairs).
  - Bash: Shebangs, error handling (`set -e`), comments for each step.
  - Focus on TA: Scripts must handle Bitcoin-specific elements rigorously (e.g., OHLCV parsing in hybrid_forecast.py, multimodal prompts in test_ta_prompt.py like "Analyze 1h BTC chart for RSI at $66k").
  - Generalization: Add flags (e.g., --asset_type) for broader use.
- **Testing**:
  - Run `pytest tests/` locally; ensure 100% coverage for new features (e.g., test_rag.py asserts relevance scores >0.18 for TA queries).
  - Validate on Synthetic Data: Use examples/btc_dataset_sample (e.g., 100 charts/CSVs); confirm RAG retrieves timestamped entries without leaks.
  - Hardware Check: Test on AMD (ROCm with GFX override); note VRAM usage (<6GB for inference); generalize to CPU.
  - Rollout Simulation: End-to-end via run_demo.sh; measure time (e.g., fine-tuning <4 hours on 16GB VRAM).
- **Documentation**:
  - Update docs/ (e.g., add to fine-tuning.md for new LoRA params).
  - Bitcoin TA Emphasis: Include examples like "Adapt for stock TA by modifying metadata schema to include asset tickers."
- **Integrations**: If drawing from cross-referenced repos (e.g., llmware's add_files for RAG), attribute in comments and docs/extensions.md. Ensure compatibility (e.g., GGUF for Ollama).
- **PR Template**: Use `.github/PULL_REQUEST_TEMPLATE.md`; include before/after benchmarks (e.g., RAG query latency).
- **Review Process**: Maintainers review within 48 hours; address feedback rigorously. Merge requires CI pass and 1 approval.

### 3. Feature Suggestions
- **Core Alignment**: Prioritize enhancements to RAG (e.g., hybrid semantic/metadata search for price highs), fine-tuning (e.g., QLoRA for lower VRAM), or hybrids (e.g., integrate statsmodels for advanced TA indicators).
- **Extensions**: Welcome modules for real-time data (proxies only, no direct API keys), multimodal improvements (e.g., OCR for chart labels via integrations like anything-llm), or domain generalizations (e.g., beyond Bitcoin to altcoins via asset_type).
- **Dataset Contributions**: Synthetic TA generators (e.g., Matplotlib scripts for candlesticks); ensure CC0 licensing; auto-annotation tools for broader appeal.

### 4. Setup for Development
- Clone and setup: Follow README quickstart.
- Dependencies: Use requirements.txt; add new ones sparingly (maintain <2GB pip install).
- Isolation: Develop in /mnt/games_storage equivalent; monitor du -sh for <35GB.
- Iteration: Test iterations on personal datasets; share anonymized results in issues/PRs.

### 5. Maintainers and Credits
- Primary: tc (@toxic2040) (e.g., for Bitcoin TA focus).
- Credits: Acknowledge inspirations (e.g., Unsloth for LoRA, llmware for RAG) in commits/docs.

By contributing, you help refine the design and rollout of this local LLM hybrid, making it more robust for Bitcoin TA on personal datasets while broadening appeal. Let's build rigorously! 🚀
