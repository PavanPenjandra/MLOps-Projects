import os
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


def build_retriever(text_path: str):
    loader = TextLoader(text_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(split_docs, embeddings, persist_directory="./chroma_store")
    db.persist()
    return db.as_retriever(search_kwargs={"k": 3})


def main():
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Set OPENAI_API_KEY before running this script.")

    retriever = build_retriever("./workshop_samples/sample_doc.txt")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0.1, model_name="gpt-4o-mini"),
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )

    print("Start a simple conversational QA session. Type 'exit' to quit.")
    while True:
        question = input("Ask a question: ")
        if question.strip().lower() == "exit":
            break
        result = qa_chain({
            "question": question,
            "chat_history": memory.load_memory_variables({"chat_history": None})["chat_history"],
        })

        print("\nAnswer:\n", result["answer"])
        print("\nSources:")
        for doc in result["source_documents"]:
            print(f"- {doc.metadata.get('source', 'unknown')}\n  {doc.page_content[:200].strip()}\n")


if __name__ == "__main__":
    main()
