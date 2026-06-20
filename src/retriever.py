import chromadb

from embeddings import embedding_model


client = chromadb.PersistentClient(path="database")

collection = client.get_collection("company_documents")


def retrieve(query, n_results=3):

    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    retrieved_chunks = []

    for i in range(len(results["ids"][0])):

        chunk = {
            "id": results["ids"][0][i],
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        }

        retrieved_chunks.append(chunk)

    return retrieved_chunks

if __name__ == "__main__":
    results = retrieve("What is the leave policy?")

    for chunk in results:
        print(chunk)
        print("-" * 80)