# Chains, Agents, and RAG

This document covers common LangChain workflows and architecture patterns.

## 1. Building chains

Chains connect prompts, LLMs, and data.

### LLMChain
Use `LLMChain` when you need one prompt and one model call.

### SimpleSequentialChain
Chain multiple `LLMChain` objects and pass output from one into the next.

### ConversationalRetrievalChain
Combines a retriever with an LLM to answer questions in a conversational way.

## 2. Retrieval-Augmented Generation (RAG)

RAG adds relevant source text to a model prompt before generation.

### Steps:
1. Load documents.
2. Split into chunks.
3. Create embeddings.
4. Store vectors in a vector database.
5. Retrieve relevant chunks for user queries.
6. Pass retrieved text to the model.

This pattern improves accuracy and reduces hallucinations.

## 3. Agent workflows

Agents are dynamically-planned systems that choose tools.

### Tool design
Each tool needs:
- a clear name
- a concise description
- a function that performs the action

### Agent types
- `ZERO_SHOT_REACT_DESCRIPTION`: general agent that reasons with tool descriptions.
- `STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION`: chat-based agent with stronger structure.

### Example pattern
1. Define tools.
2. Create an LLM.
3. Initialize an agent.
4. Call `agent.run(user_input)`.

## 4. When to use chains vs agents

- Use chains for deterministic workflows and fixed steps.
- Use agents for open-ended tasks that need tool selection or planning.

## 5. Combining RAG with agents

A powerful pattern is an agent that uses a RAG retriever as a tool.
This lets the model search documents and then decide whether to answer or call another tool.

## 6. Evaluation approaches

- test prompts against known answers
- validate retrieved context
- compare model responses with reference outputs
- track tool selection behavior in agents

## 7. Practical example

See `workshop_samples/sample_6_conversational_rag.py` for a conversational RAG demo.
See `workshop_samples/sample_7_agent_with_tools.py` for an agent composition example.
