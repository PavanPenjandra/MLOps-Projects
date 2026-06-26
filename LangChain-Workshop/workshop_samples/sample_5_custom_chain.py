import os
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SimpleSequentialChain


def build_prompt():
    system_template = "You are an expert LangChain instructor."
    human_template = (
        "Rewrite the following text for clarity and then summarize it in one paragraph:\n\n" 
        "{input_text}"
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template),
    ])
    return prompt


def main():
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Set OPENAI_API_KEY before running this script.")

    input_text = (
        "LangChain helps connect LLMs to data, tools, and workflows so that developers can build "
        "apps with reasoning, memory, and retrieval."
    )

    prompt = build_prompt()
    llm = ChatOpenAI(temperature=0.2, model_name="gpt-4o-mini")

    rewrite_chain = LLMChain(llm=llm, prompt=prompt, output_key="rewritten")
    summary_chain = LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("You are an expert summarizer."),
            HumanMessagePromptTemplate.from_template(
                "Summarize this text in one concise paragraph:\n\n{rewritten}"
            ),
        ]),
        output_key="summary",
    )

    overall_chain = SimpleSequentialChain(chains=[rewrite_chain, summary_chain], verbose=True)
    result = overall_chain.run(input_text)

    print("Final output:\n", result)


if __name__ == "__main__":
    main()
