import chromadb
from langchain.schema.document import Document

class ChromaClass:
    def __init__(self, chunks: list[Document]):
        self.chunks = chunks
        self.chroma_client = chromadb.PersistentClient(path="vectordb")
        self.collection = self.chroma_client.get_or_create_collection(name="my_collection")
        existing_ids = set(self.collection.get()['ids'])

        for i, chunk in enumerate(self.chunks):
            source = chunk.metadata.get('source', 'unknown')
            page = chunk.metadata.get('page', '0')
            idx = f"id:{source}:{page}:{i}"

            if idx not in existing_ids:
                self.collection.add(
                    documents=[chunk.page_content],
                    metadatas=[chunk.metadata],
                    ids=[idx],
                )
        
    def search(self, query: str, top_k: int = 2):

        results = self.collection.query(query_texts=[query], n_results=top_k)
        
        docs: list[Document] = []
        for content, metadata in zip(results['documents'][0], results['metadatas'][0]):
            docs.append(Document(page_content=content, metadata=metadata))
        
        return docs