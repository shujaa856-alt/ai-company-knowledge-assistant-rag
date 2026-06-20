import chromadb

client = chromadb.PersistentClient(path="database")

collection = client.get_collection(
    name="company_documents"
)

data = collection.get(
    include=["embeddings", "documents", "metadatas"]
)

print("=" * 50)
print("Total Records:", collection.count())

print("\nFirst ID:")
print(data["ids"][0])

print("\nFirst Document:")
print(data["documents"][0])

print("\nFirst Metadata:")
print(data["metadatas"][0])

print("\nEmbedding Dimensions:")
print(len(data["embeddings"][0]))