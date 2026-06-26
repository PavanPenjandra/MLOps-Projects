import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma


def main():
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Set OPENAI_API_KEY before running this script.")

    loader = TextLoader("./workshop_samples/sample_doc.txt")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_store")
    db.persist()

    query = "What is the best way to structure a LangChain workshop?"
    results = db.similarity_search(query, k=4)

    print("Top vector store results for query:\n", query)
    for i, doc in enumerate(results, start=1):
        print(f"\nResult {i}:\n{doc.page_content[:400].strip()}\n")


if __name__ == "__main__":
    main()
