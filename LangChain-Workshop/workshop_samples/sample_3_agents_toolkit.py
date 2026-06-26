import os
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType


def calculator_tool(query: str) -> str:
    try:
        return str(eval(query, {"__builtins__": {}}, {}))
    except Exception as exc:
        return f"Calculator error: {exc}"


def reverse_text_tool(text: str) -> str:
    return text[::-1]


def main():
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Set OPENAI_API_KEY before running this script.")

    tools = [
        Tool(
            name="calculator",
            func=calculator_tool,
            description="Performs safe arithmetic calculations, like 24 + 18 / 3.",
        ),
        Tool(
            name="reverse_text",
            func=reverse_text_tool,
            description="Reverses text for demonstration and debugging purposes.",
        ),
    ]

    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    prompt = "Calculate 12 * 7 and then reverse the text of the result."
    response = agent.run(prompt)
    print("\nAgent response:\n", response)


if __name__ == "__main__":
    main()
