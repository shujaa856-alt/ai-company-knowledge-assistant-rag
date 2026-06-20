from loader import load_documents
from splitter import split_documents
from embeddings import create_embeddings
from vectordb import reset_collection, store_embeddings


def ingest():
    print("Preparing database...")
    reset_collection()

    print("Loading documents...")
    documents = load_documents()

    print(f"Loaded {len(documents)} documents.")

    print("\nSplitting documents...")
    chunks = split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    print("\nGenerating embeddings...")
    embeddings = create_embeddings(chunks)

    print(f"Generated {len(embeddings)} embeddings.")

    print("\nStoring embeddings in ChromaDB...")
    store_embeddings(chunks, embeddings)

    print("\nKnowledge base created successfully!")

if __name__ == "__main__":
    ingest()