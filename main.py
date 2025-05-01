from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chroma_class import chroma_search
from faiss_class import faiss_search
from ollama import Client

client = Client(
    host='http://localhost:11434',
)

DATA_PATH = "data"

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def main():
    docs = load_documents()
    chunks = split_documents(docs)

    while True:
        user_input = input("Please enter your question: \n")
        input_text = input("Please choose one of the following: 1. faiss 2. chroma\n")
        if input_text.strip().lower() in ("faiss", "1"):
            results = faiss_search(chunks, user_input, top_k=2)
            break
        elif input_text.strip().lower() in ("chroma", "2"):
            results = chroma_search(chunks, user_input, top_k=2)
            break
        else:
            print("Invalid choice. Please try again.")

    response = client.chat(
        model='qwen2.5',
        messages=[
            {
                'role': 'system',
                'content': 'You are a helpful assistant that uses provided documents to answer questions.',
            },
            {
                'role': 'user',
                'content': user_input,
            },
            {
                'role': 'system',
                'content': f'Documents: {results}'
            }
        ],
    )

    print(response)

if __name__ == "__main__":
    main()
