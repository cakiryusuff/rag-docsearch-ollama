import chromadb
from langchain.schema.document import Document

def chroma_search(chunks: list[Document], query: str, top_k: int = 2):
    chroma_client = chromadb.PersistentClient(path="vectordb")
    collection = chroma_client.get_or_create_collection(name="my_collection")
    existing_ids = set(collection.get()['ids'])

    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get('source', 'unknown')
        page = chunk.metadata.get('page', '0')
        idx = f"id:{source}:{page}:{i}"

        if idx not in existing_ids:
            collection.add(
                documents=[chunk.page_content],
                metadatas=[chunk.metadata],
                ids=[idx],
            )

    results = collection.query(query_texts=[query], n_results=top_k)
    
    docs: list[Document] = []
    for content, metadata in zip(results['documents'][0], results['metadatas'][0]):
        docs.append(Document(page_content=content, metadata=metadata))
    
    return docs