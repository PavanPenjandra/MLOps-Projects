# LangChain Workshop

This project is a complete LangChain tutorial designed for learners who want both a clear introduction and a deep, practical understanding of the framework.

## What’s included

- `beginner_tutorial.md`: core LangChain concepts and first workflows.
- `advanced_tutorial.md`: advanced patterns, agents, tools, RAG, and production guidance.
- `docs/`: detailed LangChain architecture, component explanations, workflow patterns, and best practices.
- `workshop_samples/`: runnable Python examples demonstrating real LangChain use cases.
- `.env.example`: example environment variables.
- `requirements.txt`: dependencies for running the samples.

## Project structure

- `docs/01_overview.md`
- `docs/02_core_components.md`
- `docs/03_chains_agents_rag.md`
- `docs/04_production.md`
- `workshop_samples/sample_1_basic_chat.py`
- `workshop_samples/sample_2_document_qa.py`
- `workshop_samples/sample_3_agents_toolkit.py`
- `workshop_samples/sample_4_vectorstore_retrieval.py`
- `workshop_samples/sample_5_custom_chain.py`
- `workshop_samples/sample_6_conversational_rag.py`
- `workshop_samples/sample_7_agent_with_tools.py`

## Learning path

1. Read `docs/01_overview.md` and `beginner_tutorial.md`.
2. Run the beginner examples in `workshop_samples/`.
3. Study `docs/02_core_components.md` and `docs/03_chains_agents_rag.md`.
4. Try the advanced examples and build your own workflows.
5. Read `docs/04_production.md` to learn how to scale, test, and deploy LangChain applications.

## Setup

```bash
cd LangChain-Workshop
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set your API key:

```bash
copy .env.example .env
```

## Running samples

- `python workshop_samples/sample_1_basic_chat.py`
- `python workshop_samples/sample_2_document_qa.py`
- `python workshop_samples/sample_3_agents_toolkit.py`
- `python workshop_samples/sample_4_vectorstore_retrieval.py`
- `python workshop_samples/sample_5_custom_chain.py`
- `python workshop_samples/sample_6_conversational_rag.py`
- `python workshop_samples/sample_7_agent_with_tools.py`

## Notes

- Set `OPENAI_API_KEY` or another provider API key before running the examples.
- Use the docs to understand why each pattern works, not just how to run it.
- Extend the sample scripts to make the project your own.
