from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_documents(documents):
    """
    Split documents into smaller chunks.

    Args:
        documents (List[Document]):
            Documents returned by load_documents().

    Returns:
        List[Document]:
            Chunked LangChain Document objects.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    return chunks