import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI


def build_retriever(text_path: str):
    loader = TextLoader(text_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    split_docs = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(split_docs, embeddings, persist_directory="./chroma_store")
    vectorstore.persist()
    return vectorstore.as_retriever(search_kwargs={"k": 3})


def main():
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Set OPENAI_API_KEY before running this script.")

    os.makedirs("./chroma_store", exist_ok=True)
    retriever = build_retriever("./workshop_samples/sample_doc.txt")

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0.0, model_name="gpt-4o-mini"),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    question = "What are the two main benefits of using LangChain for production apps?"
    result = qa_chain(question)

    print("Question:\n", question)
    print("\nAnswer:\n", result["result"])
    print("\nSource documents:\n")
    for doc in result["source_documents"]:
        print(f"- {doc.metadata.get('source')} | {doc.page_content[:150].strip()}...")


if __name__ == "__main__":
    main()
