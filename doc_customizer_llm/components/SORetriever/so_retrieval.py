# ==========================================================================================================================
# IMPORT DEPENDENCIES
# ==========================================================================================================================

import os
import pandas as pd
from dotenv import load_dotenv

COLOR = {
    "HEADER": "\033[95m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "RED": "\033[91m",
    "ENDC": "\033[0m",
}

from langchain_openai import OpenAIEmbeddings
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_loaders import DataFrameLoader
from langchain.retrievers.document_compressors import CohereRerank
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load custom modules
from .lib.api_funcs import query_so

# ==========================================================================================================================
# LOAD API KEYS FROM THE .env FILE
# ==========================================================================================================================

load_dotenv(dotenv_path="/Users/sharukat/Documents/ResearchYU/Code/doc-customizer-llm/doc_customizer_llm/.env")
os.environ["VOYAGE_API_KEY"] = os.getenv("VOYAGE_API_KEY")   # Claude LLM API Key
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")   # LangChain LLM API Key (To use LangSmith)

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "langchain-stackoverflow-search"


# ==========================================================================================================================
# MAIN CODE
# ==========================================================================================================================

embedding_function = VoyageAIEmbeddings(model="voyage-large-2", batch_size=32)

def retrieve_relevant_from_so(question_intent, question_body):
    print(f"{COLOR['BLUE']}🚀: EXECUTING Q&A RETRIEVER: Searching for relevant SO Q&As with accepted answers.{COLOR['ENDC']}")
    res = query_so(question_intent)
    results = []

    if res:
        df = pd.DataFrame(res)
        print(f"\t➡️ Relevant Stack Overflow Q&A Count: {df.shape[0]}")

        loader = DataFrameLoader(df, page_content_column="Answer")
        answers = loader.load()

        # parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=10)
        vectorstore = Chroma(
            collection_name="answers",
            embedding_function=embedding_function
        )

        # The storage layer for the full answer
        store = InMemoryStore()

        full_answer_retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            # parent_splitter=parent_splitter,
        )

        full_answer_retriever.add_documents(answers, ids=None)
        compressor = CohereRerank(model='rerank-english-v2.0')
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=full_answer_retriever , 
        )

        reranked_answers = compression_retriever.get_relevant_documents(query=question_body)
        for index, answer in enumerate(reranked_answers):
           if answer.metadata['relevance_score'] > 0.6:
              results.append(answer)
        print(f"{COLOR['GREEN']}✅: EXECUTION COMPLETED{COLOR['ENDC']}\n")

        if len(results) < 5:
            return results
        else:
            return results[:5]
    else:
        print(f"{COLOR['RED']}✅: ENO RELEVANT STACK OVERFLOW Q&A FOUND.{COLOR['ENDC']}\n")
        return None