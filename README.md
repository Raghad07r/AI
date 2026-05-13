# Project: RAG 

## Overview
This project integrates all three lab tracks into a single production-ready pipeline.

## Structure
```
project_starter/
├── project_rag_application.ipynb   ← Main notebook (complete the TODOs)
├── tests/
│   ├── __init__.py
│   └── checks.py                   ← Validation functions
└── README.md
```

## Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install uv
uv pip install litellm python-dotenv chromadb numpy langchain-text-splitters \
               sentence-transformers requests beautifulsoup4 deepeval tiktoken

# Create .env file
echo "OPENROUTER_API_KEY=sk-or-..." > .env
```

## How to Run
Open `project_rag_application.ipynb` in Jupyter and run cells top-to-bottom.
Complete each `# TODO` block — the `checks` module will validate your work.

## Exercises Summary
| Exercise | Skill | What you implement |
|----------|-------|--------------------|
| 1 | Ingestion | Extract `.txt` → chunk → embed → store |
| 2 | Retrieval | Metadata-filtered vector search |
| 3 | Observability | Loop detection in the RAG pipeline |
| 4 | Cost | Pre-flight ingestion cost estimator |

## Phases
1. **Document Model** — `Document` dataclass + dispatcher
2. **Chunking & Embedding** — `RecursiveChunker` + `EmbeddingGenerator`
3. **Vector Store** — `VectorStoreManager` (ChromaDB HNSW)
4. **Observability** — `AgentTracer` + `ToolCallRecord`
5. **Token Budget** — `check_token_budget()` with tiktoken
6. **Traced RAG** — `TracedRAGService` (Retrieve → Augment → Generate)
7. **Multi-Agent Router** — `QueryRouter` (classify → rewrite → route)
8. **End-to-End Run** — Wire all components and query the system
9. **Evaluation** — DeepEval `TaskCompletionMetric`
