from sentence_transformers import SentenceTransformer
from langchain.schema import Document
import faiss

def faiss_search(chunks: list[Document], query: str, top_k: int = 2, embedding_model=None):
    if embedding_model is None:
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [doc.page_content for doc in chunks]
    embeddings = embedding_model.encode(texts, show_progress_bar=False).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    query_embedding = embedding_model.encode([query]).astype('float32')

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        doc = chunks[idx]
        doc.metadata["similarity"] = float(dist)
        results.append(doc)

    return results
