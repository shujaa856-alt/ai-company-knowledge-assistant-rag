import chromadb

# Create the persistent client once
client = chromadb.PersistentClient(path="database")


def reset_collection():
    """
    Delete the existing collection (if it exists)
    and create a fresh one.
    """

    try:
        client.delete_collection("company_documents")
        print("Existing collection deleted.")
    except Exception:
        print("No existing collection found. Creating a new one.")

    client.get_or_create_collection(
        name="company_documents"
    )


def store_embeddings(chunks, embeddings):

    collection = client.get_or_create_collection(
        name="company_documents"
    )

    ids = []
    documents = []
    metadatas = []

    for index, chunk in enumerate(chunks):
        ids.append(f"chunk_{index}")
        documents.append(chunk.page_content)
        metadatas.append(chunk.metadata)

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"Stored {len(ids)} chunks in ChromaDB.")