# ==========================================================================================================================
# IMPORT DEPENDENCIES
# ==========================================================================================================================
import os
import pandas as pd
from lib.config import COLOR
from dotenv import load_dotenv

from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
# from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_loaders import DataFrameLoader
from langchain.retrievers.document_compressors import CohereRerank
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import CohereEmbeddings
from langchain_cohere import CohereEmbeddings, CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_voyageai import VoyageAIEmbeddings


# Load custom modules
from lib.api import stackexchange

load_dotenv(dotenv_path="/Users/sharukat/Documents/ResearchYU/Code/doc-customizer-llm/doc_customizer_llm/.env")
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")   # Claude LLM API Key
os.environ["VOYAGE_API_KEY"] = os.getenv("VOYAGE_API_KEY") 


# ==========================================================================================================================
# MAIN CODE
# ==========================================================================================================================

def retrieval(title, intent):
    print(f"{COLOR['BLUE']}🚀: EXECUTING Q&A RETRIEVER: Searching for relevant SO Q&As with accepted answers.{COLOR['ENDC']}")
    res = stackexchange(title)
    answers_pagecontent = set()
    qids = set()
    urls = set()

    if res:
        df = pd.DataFrame(res)
        print(f"\t➡️ Relevant Stack Overflow Q&A Count: {df.shape[0]}")

        loader = DataFrameLoader(df, page_content_column="Answer")
        answers = loader.load()
        # embedding_function = CohereEmbeddings(model="embed-english-v3.0")
        embedding_function = VoyageAIEmbeddings(model="voyage-large-2-instruct" )

        text_splitter = SemanticChunker(embeddings=embedding_function, breakpoint_threshold_type="percentile")

        # parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        # child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=10)

        # vectorstore = Chroma(
        #     collection_name="answers",
        #     embedding_function=embedding_function
        # )

        # # The storage layer for the full answer
        # store = InMemoryStore()

        # full_answer_retriever = ParentDocumentRetriever(
        #     vectorstore=vectorstore,
        #     docstore=store,
        #     child_splitter=text_splitter,
        #     # parent_splitter=parent_splitter,
        # )

        splits = text_splitter.split_documents(answers)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_function)
        full_answer_retriever = vectorstore.as_retriever(search_kwargs={"k": len(splits)})

        full_answer_retriever.add_documents(answers, ids=None)
        compressor = CohereRerank(model='rerank-english-v2.0')
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=full_answer_retriever , 
        )

        reranked_answers = compression_retriever.invoke(input=intent)
        for index, answer in enumerate(reranked_answers):
           if answer.metadata['relevance_score'] > 0.6:
              answers_pagecontent.add(answer.page_content)
              urls.add(answer.metadata['URL'])
              qids.add(answer.metadata['QuestionId'])

        print(f"\tNumber of URLs: {len(urls)}")
        print(f"{COLOR['GREEN']}✅: EXECUTION COMPLETED{COLOR['ENDC']}\n")

        results = {"answers":list(answers_pagecontent), "question_ids":qids, "urls":urls}

        return results

        # if len(results) < 5:
        #     return results
        # else:
        #     return results[:5]
    else:
        print(f"{COLOR['RED']}✅: NO RELEVANT STACK OVERFLOW QUESTIONS WITH ACCEPTED ANSWERS WERE FOUND.{COLOR['ENDC']}\n")
        return None
    


# Test
# title = "How to create output_signature for tensorflow.dataset.from_generator"
# intent = """
# The user is seeking to understand how to correctly create an output_signature for a TensorFlow dataset generated from a generator function, specifically when dealing with variable row sizes in the data.
# """

# so_answers, ids, urls = retrieval(title, intent)
# print(f"Relevant Chunks: \n{'='*20}\n{so_answers}\n")
# print(f"Questions IDs: {ids}\n")
# print(f"URLs: \n{'='*20}\n{urls}")
