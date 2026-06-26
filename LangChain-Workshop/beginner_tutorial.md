# LangChain Beginner Tutorial

Welcome to the LangChain beginner workshop. This guide covers the core building blocks and helps you write your first working chains.

## 1. What is LangChain?

LangChain is a framework for building applications with large language models. It helps you connect LLMs to data, logic, and external tools.

Core concepts:
- **LLM**: the language model that generates text.
- **Prompt**: the instruction sent to the model.
- **Chain**: a sequence of actions connecting prompts, models, and data.
- **Memory**: state over multiple interactions.
- **Agent**: a decision-making workflow that uses tools.

## 2. Your first chat app

Open `workshop_samples/sample_1_basic_chat.py` and run it.

Key ideas:
- `ChatOpenAI` creates the chat model.
- `HumanMessage` wraps user input.
- `model.predict_messages()` returns the model response.

## 3. Text splitting and embedding

Useful when your app needs to answer questions from documents.

Steps:
1. Load text.
2. Split into smaller chunks.
3. Embed chunks with an embeddings model.
4. Store vectors in a vector database.
5. Use retrieval to answer questions.

See `workshop_samples/sample_2_document_qa.py`.

## 4. Building a simple retrieval QA chain

A retrieval QA chain combines:
- a retriever (`Chroma` / vectorstore search)
- an LLM prompt that answers with context

This pattern is the foundation for many LangChain apps.

## 5. Common beginner exercises

- Change the prompt to make the model answer in bullet points.
- Load a new document and see how retrieval changes.
- Add a second question and keep the context.
- Try a different model temperature.

## 6. Moving from beginner to intermediate

Practice these patterns:
- `LLMChain`
- `SequentialChain`
- `ConversationalRetrievalChain`
- `Memory` for multi-turn sessions

The next tutorial covers these concepts in detail.
