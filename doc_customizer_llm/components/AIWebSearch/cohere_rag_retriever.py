# ==========================================================================================================================
# IMPORT DEPENDENCIES
# ==========================================================================================================================

from langchain_voyageai import VoyageAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.retrievers import CohereRagRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain_community.chat_models import ChatCohere
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
import os
from dotenv import load_dotenv

COLOR = {
    "HEADER": "\033[95m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "RED": "\033[91m",
    "ENDC": "\033[0m",
}

# LANGCHAIN MODULES


# ==========================================================================================================================
# LOAD API KEYS FROM THE .env FILE
# ==========================================================================================================================

load_dotenv(dotenv_path="../../.env")
os.environ["COHERE_API_KEY"] = os.getenv(
    "COHERE_API_KEY"
)  # Cohere Command-R LLM API Key
os.environ["LANGCHAIN_API_KEY"] = os.getenv(
    "LANGCHAIN_API_KEY"
)  # LangChain LLM API Key (To use LangSmith)
os.environ["VOYAGE_API_KEY"] = os.getenv("VOYAGE_API_KEY")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "langchain-websearch"


# ==========================================================================================================================
# MAIN CODE
# ==========================================================================================================================


def relevant_context_retriever(query: str):
    print(
        f"{COLOR['BLUE']}🚀: EXECUTING RETRIEVER: Cohere RAG Retriever{COLOR['ENDC']}"
    )
    urls = []
    docs = []
    model_response = []
    context = []
    rag = CohereRagRetriever(
        llm=ChatCohere(model="command-r"), connectors=[{"id": "web-search"}]
    )

    compressor = CohereRerank(model="rerank-english-v2.0", top_n=5)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=rag,
    )

    reranked_docs = compression_retriever.get_relevant_documents(query=query)
    for _, doc in enumerate(reranked_docs):
        if "type" in doc.metadata:
            model_response.append(doc.page_content)
        else:
            if doc.metadata["relevance_score"] > 0.6:
                urls.append(doc.metadata["url"])
                docs.append(doc)

    embedding_function = VoyageAIEmbeddings(
        model="voyage-large-2", batch_size=32)

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000, chunk_overlap=50)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=10)
    vectorstore = Chroma(
        collection_name="context", embedding_function=embedding_function
    )

    # The storage layer for the full answer
    store = InMemoryStore()

    context_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    context_retriever.add_documents(docs, ids=None)
    compressor = CohereRerank(model="rerank-english-v2.0", top_n=10)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=context_retriever,
    )

    relevant_context = compression_retriever.get_relevant_documents(
        query=query)
    for _, answer in enumerate(relevant_context):
        if answer.metadata["relevance_score"] > 0.6:
            context.append(answer.page_content)

    print(f"\t➡️ Relevant Parent Doc Count: {len(context)}")

    print(f"{COLOR['GREEN']}✅: EXECUTION COMPLETED{COLOR['ENDC']}\n")
    return model_response, context, urls
