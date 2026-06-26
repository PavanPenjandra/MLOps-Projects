# LangChain Core Components

This document explains the foundational pieces in LangChain.

## 1. LLMs and chat models

An LLM is the text generation engine. In LangChain, common classes are:
- `OpenAI` for single-turn text completion.
- `ChatOpenAI` for chat-style models.

Example:
```python
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)
```

## 2. Prompt templates

Prompt templates let you define reusable, parameterized prompts.
- `PromptTemplate`: plain text prompts.
- `ChatPromptTemplate`: chat-style system/user messages.

Example:
```python
from langchain.prompts import PromptTemplate
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in layman terms."
)
```

## 3. Messages and schema

For chat models, LangChain uses message classes such as:
- `HumanMessage`
- `SystemMessage`
- `AIMessage`

These help structure multi-turn conversation history cleanly.

## 4. Chains

A chain is a sequence of steps executed in order.
- `LLMChain`: single prompt and LLM call.
- `SequentialChain`: run multiple chains in sequence.
- `SimpleSequentialChain`: simpler sequencing of chains.

Example:
```python
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)
```

## 5. Memory

Memory stores conversation state across user turns.
- `ConversationBufferMemory` stores all conversation history.
- `ConversationSummaryMemory` summarizes prior turns.
- `VectorStoreRetrieverMemory` stores memory as embeddings.

Memory is essential for building assistants that remember context.

## 6. Tools and agents

Tools enable an agent to perform actions beyond text generation.
- `Tool`: wraps Python functions or external APIs.
- `initialize_agent()`: creates an agent from tools and an LLM.

Agents choose which tool to call based on the user request.

## 7. Vectors and retrieval

Retrieval uses embeddings and vector stores to answer questions from documents.
- `Embeddings`: convert text into numeric vectors.
- `VectorStore`: stores vectors for similarity search.
- `Retriever`: searches the vector store for relevant chunks.

Common vector stores include `Chroma`, `FAISS`, and `Weaviate`.

## 8. Common patterns

- retrieval-augmented generation (RAG)
- conversational QA
- tool-based reasoning
- multi-step chains
- prompt refinement

These building blocks combine to create sophisticated LangChain apps.
