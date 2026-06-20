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

    return results

if __name__ == "__main__":
    results = retrieve("What is the leave policy?")

    print("Documents:")
print(results["documents"])

print("\nMetadata:")
print(results["metadatas"])

print("\nDistances:")
print(results["distances"])

print("\nIDs:")
print(results["ids"])