import glob
from typing import List, Dict

from langchain.document_loaders import UnstructuredURLLoader, UnstructuredFileLoader, PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.vectorstores import Chroma

TOKENS_PER_CHUNK = 2000
CHUNK_OVERLAP = 300
PERSIST_DIRECTORY = "chroma_db"
EMBEDDINGS_FUNC = OpenAIEmbeddings()
SEARCH = DuckDuckGoSearchAPIWrapper()

"""
docsearch._collection.get(where={'source': './knowledge_base\\648-1-39928-1-10-20220823.pdf'})

res = docsearch.similarity_search("Causal relation of sea ice with sea surface temperature",
                            filter={
                                "$and": [
                                    {"source": "./knowledge_base\\1520-0442-jcli-d-16-0136.1.pdf"},
                                    {"page": 2}
                                    ]
                            })
"""


def build_vector_store_from_folder(folder_path, persist_directory=PERSIST_DIRECTORY, tokens_per_chunk=TOKENS_PER_CHUNK, chunk_overlap=CHUNK_OVERLAP) -> Chroma:
    documents = read_files_from_folder(folder_path)
    documents = split_documents(documents, tokens_per_chunk, chunk_overlap)
    vector_store = create_vector_store(persist_directory)
    vector_store.add_documents(documents)
    vector_store.persist()
    return vector_store


def create_vector_store(persist_directory=PERSIST_DIRECTORY) -> Chroma:
    vector_store = Chroma(embedding_function=EMBEDDINGS_FUNC, persist_directory=persist_directory)
    vector_store.persist()
    return vector_store


def load_vector_store(persist_directory=PERSIST_DIRECTORY) -> Chroma:
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=EMBEDDINGS_FUNC)
    return vector_store


def read_files_from_folder(folder_path):
    pdf_files = glob.glob(f"{folder_path}/*.pdf")
    txt_files = glob.glob(f"{folder_path}/*.txt")
    print(f"Found {len(pdf_files)} pdf files and {len(txt_files)} txt files")

    documents = []
    for pdf_file in pdf_files:
        try:
            loader = UnstructuredFileLoader(pdf_file, mode="single")
            docs = loader.load()
        except Exception as e:
            print(e)
            loader = PyPDFLoader(pdf_file)
            docs = loader.load()

        documents.extend(docs)

    for txt_file in txt_files:
        with open(txt_file, "r") as file:
            page_content = file.read()

            document = Document(page_content=page_content, metadata={"source": txt_file, "page": 0})
            documents.append(document)

    return documents


def split_documents(documents: List[Document], tokens_per_chunk=TOKENS_PER_CHUNK, chunk_overlap=CHUNK_OVERLAP) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=tokens_per_chunk, chunk_overlap=chunk_overlap, add_start_index=True)
    print("before split", len(documents))
    documents = text_splitter.split_documents(documents)
    print("after split", len(documents))
    return documents



#########################
# Automatic knowledge base creation from web searches
#########################

def search_web(query: str, num_results: int = 3) -> List[Dict[str, str]]:
    results = SEARCH.results(query, num_results=num_results)
    return results


def read_web_results(query: str, results: List[Dict[str, str]]) -> List[Document]:
    urls = [result["link"] for result in results]
    documents = UnstructuredURLLoader(urls).load()
    [document.metadata.update({"query": query}) for document in documents]
    return documents


def create_knowledge_base_from_web(query: str, vector_store: Chroma, num_results: int = 3):
    results = search_web(query, num_results=num_results)
    documents = read_web_results(query, results)
    documents = split_documents(documents)
    vector_store.add_documents(documents)
    vector_store.persist()


def build_knowledge_base_from_web(Xs: List[str], Ys: List[str], vector_store: Chroma):
    if len(Xs) != len(Ys):
        raise ValueError("Xs and Ys must have the same length")

    for X, Y in zip(Xs, Ys):
        print(X, Y)
        queries = [X, Y, f"Can a change in {X} lead to a change in {Y}?", f"Can a change in {Y} lead to a change in {X}?"]
        for query in queries:
            urls = search_web(query)
            documents = read_web_results(query, urls)
            documents = split_documents(documents)
            vector_store.add_documents(documents)


def query_knowledge_base_with_query_filter(query: str, vector_store: Chroma, k: int = 7) -> List[Document]:
    documents = vector_store.similarity_search(query, k=k, filter={"query": query})
    return documents


