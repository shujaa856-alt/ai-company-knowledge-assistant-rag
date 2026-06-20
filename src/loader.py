from langchain_community.document_loaders import PyPDFDirectoryLoader


def load_documents():
    """
    Load all PDF documents from the data/documents directory.

    Returns:
        List[Document]: A list of LangChain Document objects.
    """
    loader = PyPDFDirectoryLoader("data/documents")

    documents = loader.load()

    return documents