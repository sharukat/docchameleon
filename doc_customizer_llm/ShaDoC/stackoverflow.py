# ==========================================================================================================================
# IMPORT DEPENDENCIES
# ==========================================================================================================================

import pandas as pd
from lib.config import COLOR
from dotenv import load_dotenv

from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_loaders import DataFrameLoader
from langchain.retrievers.document_compressors import CohereRerank
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load custom modules
from lib.api import stackexchange



# ==========================================================================================================================
# MAIN CODE
# ==========================================================================================================================

def retrieval(title, body):
    print(f"{COLOR['BLUE']}🚀: EXECUTING Q&A RETRIEVER: Searching for relevant SO Q&As with accepted answers.{COLOR['ENDC']}")
    res = stackexchange(title)
    results = []

    if res:
        df = pd.DataFrame(res)
        print(f"\t➡️ Relevant Stack Overflow Q&A Count: {df.shape[0]}")

        loader = DataFrameLoader(df, page_content_column="Answer")
        answers = loader.load()

        embedding_function = VoyageAIEmbeddings(model="voyage-large-2", batch_size=32)

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

        reranked_answers = compression_retriever.get_relevant_documents(query=body)
        for index, answer in enumerate(reranked_answers):
           if answer.metadata['relevance_score'] > 0.6:
              results.append(answer)
        print(f"{COLOR['GREEN']}✅: EXECUTION COMPLETED{COLOR['ENDC']}\n")

        if len(results) < 5:
            return results
        else:
            return results[:5]
    else:
        print(f"{COLOR['RED']}✅: NO RELEVANT STACK OVERFLOW Q&A FOUND.{COLOR['ENDC']}\n")
        return None