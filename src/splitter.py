from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader =PyPDFDirectoryLoader("documents")

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Original Documents: {len(documents)}")
print(f"Total Chunks: {len(chunks)}")

print("\n========== FIRST CHUNK ==========\n")

print("Chunk Content:\n")
print(chunks[0].page_content)

print("\n-----------------------------")

print("Metadata:\n")
print(chunks[0].metadata)