from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage


def main():
    model = ChatOpenAI(temperature=0.2, model_name="gpt-4o-mini")

    user_prompt = "Explain LangChain in simple terms for a beginner."
    messages = [HumanMessage(content=user_prompt)]

    response = model.predict_messages(messages)
    print("User prompt:\n", user_prompt)
    print("\nModel response:\n", response.content)


if __name__ == "__main__":
    main()
