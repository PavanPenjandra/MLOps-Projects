# LangChain Advanced Tutorial

This advanced workshop helps you become an expert by exploring agents, tools, custom chains, production-ready architecture, and evaluation.

## 1. Expert concepts

- **Agents** decide what tool to call based on user input.
- **Tools** connect the LLM to external capabilities such as search, calculators, or APIs.
- **Custom Chains** let you create domain-specific workflows and handlers.
- **Retrieval-Augmented Generation (RAG)** combines vector search with generative answers.
- **Tool grounding** means controlling when and how the model uses external tools.

## 2. Agents and tools

Open `workshop_samples/sample_3_agents_toolkit.py`.

Important patterns:
- Use `Tool` to define each action.
- Use `initialize_agent()` to let the agent choose the tool.
- Test edge cases to ensure the model selects the correct ability.

## 3. Vector store best practices

- Use document splitting with overlap for better context.
- Store metadata for filtering and source attribution.
- Persist your vector database in production.
- Use embeddings once and reuse them for the same corpus.

See `workshop_samples/sample_4_vectorstore_retrieval.py`.

## 4. Custom chain design

Custom chains let you:
- combine multiple prompts and LLM steps
- inject domain logic between model calls
- validate and normalize output

See `workshop_samples/sample_5_custom_chain.py`.

## 5. Production considerations

- **Prompt versioning**: keep prompts in source control.
- **Safety**: add guardrails and model output validation.
- **Monitoring**: log prompts, responses, and retrieval sources.
- **Latency**: choose smaller models for fast loops.
- **Cost**: cache embeddings and reuse vector stores.

## 6. Evaluation and testing

- Write automated tests for prompt templates and chain outputs.
- Use sample prompts to validate major workflows.
- Compare model results with ground truth answers.
- Track drift by rerunning against benchmark queries.

## 7. Expert exercises

- Replace `OpenAI` with a local or alternative provider.
- Add a browser search tool or a database query tool.
- Build a `ConversationalRetrievalChain` with memory.
- Add source attribution to every answer.
- Create an app that routes different user intents to specialized chains.
