from langchain_community.document_loaders import PyPDFDirectoryLoader

loader = PyPDFDirectoryLoader("documents")

documents = loader.load()

print(f"Loaded {len(documents)} documents.")

print("\nFirst Document:\n")

print("Source:")
print(documents[0].metadata["source"])

print("\nPage:")
print(documents[0].metadata["page"])

print("\nTotal Pages:")
print(documents[0].metadata["total_pages"])

print("\nContent:")
print(documents[0].page_content)

