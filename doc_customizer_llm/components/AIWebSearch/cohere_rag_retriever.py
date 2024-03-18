# ==========================================================================================================================
# IMPORT DEPENDENCIES
# ==========================================================================================================================

import os
from dotenv import load_dotenv

# LANGCHAIN MODULES
from langchain_community.chat_models import ChatCohere
from langchain.retrievers import CohereRagRetriever


# ==========================================================================================================================
# LOAD API KEYS FROM THE .env FILE
# ==========================================================================================================================

load_dotenv(dotenv_path="/Users/sharukat/Documents/ResearchYU/Code/doc-customizer-llm/doc_customizer_llm/.env")
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")   # Cohere Command-R LLM API Key
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")   # LangChain LLM API Key (To use LangSmith)

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "langchain-websearch"


# ==========================================================================================================================
# MAIN CODE
# ==========================================================================================================================

def relevant_context_retriever(query: str):
    print("Retrieving relevant context from web using Cohere RAG Retriever....")
    urls = []
    page_content = []
    rag = CohereRagRetriever(llm=ChatCohere(model="command-r"), connectors=[{"id": "web-search"}]) 
    docs = rag.get_relevant_documents(query)
    for doc in docs[:-1]:
        urls.append(doc.metadata['url'])
        page_content.append(doc.page_content)

    print("Completed Successfully\n")
    return urls, page_content
