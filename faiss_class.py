from sentence_transformers import SentenceTransformer
from langchain.schema import Document
import faiss

class FaissClass:
    def __init__(self, chunks: list[Document], embedding_model=None):
        self.chunks = chunks
        if embedding_model is None:
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        else:
            self.embedding_model = embedding_model

        texts = [doc.page_content for doc in self.chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False).astype("float32")

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)


    def search(self, query: str, top_k: int = 2):
        
        query_embedding = self.embedding_model.encode([query]).astype('float32')
        
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            doc = self.chunks[idx]
            doc.metadata["similarity"] = float(dist)
            results.append(doc)

        return results
