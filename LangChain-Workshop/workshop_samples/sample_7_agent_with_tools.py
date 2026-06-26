import os
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType


def math_tool(query: str) -> str:
    try:
        return str(eval(query, {"__builtins__": {}}, {}))
    except Exception as exc:
        return f"Math tool error: {exc}"


def wiki_tool(query: str) -> str:
    return f"(Wiki) I would search for '{query}' and return a concise answer."


def sentiment_tool(text: str) -> str:
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    prompt = f"Analyze the sentiment of this text and return positive, negative, or neutral:\n\n{text}"
    return llm.predict(prompt)


def main():
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Set OPENAI_API_KEY before running this script.")

    tools = [
        Tool(
            name="calculator",
            func=math_tool,
            description="Perform arithmetic or simple numeric evaluation.",
        ),
        Tool(
            name="wiki_search",
            func=wiki_tool,
            description="Answer general knowledge questions with a short explanation.",
        ),
        Tool(
            name="sentiment_analyzer",
            func=sentiment_tool,
            description="Determine whether text sentiment is positive, negative, or neutral.",
        ),
    ]

    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    prompt = (
        "You are a helper agent. Use the appropriate tool when needed. "
        "If the request is purely conversational, answer directly."
    )
    print(agent.run(prompt))


if __name__ == "__main__":
    main()
