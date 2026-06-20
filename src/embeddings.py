from sentence_transformers import SentenceTransformer

# Load the embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embeddings(chunks):
    """
    Generate embeddings for document chunks.

    Args:
        chunks (List[Document]):
            Chunked LangChain Document objects.

    Returns:
        List[List[float]]:
            A list of embedding vectors.
    """

    chunk_texts = [chunk.page_content for chunk in chunks]

    embeddings = model.encode(chunk_texts)

    return embeddings.tolist()