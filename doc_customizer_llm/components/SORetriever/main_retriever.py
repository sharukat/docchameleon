# ==========================================================================================================================
# IMPORT DEPENDENCIES
# ==========================================================================================================================

import os
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_loaders import DataFrameLoader
from langchain.retrievers.document_compressors import CohereRerank
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load custom modules
import lib.api_funcs as api

# ==========================================================================================================================
# LOAD API KEYS FROM THE .env FILE
# ==========================================================================================================================

load_dotenv(dotenv_path="/Users/sharukat/Documents/ResearchYU/Code/doc-customizer-llm/doc_customizer_llm/.env")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")   # Claude LLM API Key
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")   # LangChain LLM API Key (To use LangSmith)

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "langchain-stackoverflow-search"


# ==========================================================================================================================
# MAIN CODE
# ==========================================================================================================================

embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")

def retrieve_relevant_from_so(question_title, question_body):
    print('Searching Stack Overflow ...')
    res = api.query_so(question_title)

    if res:
        df = pd.DataFrame(res)
        print(f"Total number of answers retrieved from Stack Overflow: {df.shape[0]}")

        loader = DataFrameLoader(df, page_content_column="Answer")
        answers = loader.load()

        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
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
        compressor = CohereRerank(model='rerank-english-v2.0', top_n=4)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=full_answer_retriever , 
        )

        reranked_answers = compression_retriever.get_relevant_documents(query=question_body)
        return reranked_answers
    return None


# ==========================================================================================================================
# TEST EXECUTIONS
# ==========================================================================================================================

title = "Custom initializer for get_variable"
body = '''
<p>How can one specify a custom initializer as the third argument for <code>tf.get_variable()</code>? Specifically, I have a variable <code>y</code> which I want to initialize using another (already initialized) variable <code>x</code>. </p>

<p>This is easy to do using <code>tf.Variable()</code>, just say, <code>y = tf.Variable(x.initialized_value())</code>. But I couldn't find an analog in the documentation for <code>tf.get_variable()</code>.</p>


'''

docs = retrieve_relevant_from_so(question_title=title, question_body=body)
if docs != None:
  for idx, doc in enumerate(docs):
    print(f"{'=' * 100}")
    print(f"Document Rank: {idx + 1} | Relevant Score: {doc.metadata['relevance_score']}")
    print(f"{'=' * 100}\n")
    print(f"{doc.page_content}")
    print(f"\nURL: {doc.metadata['URL']}\n\n")
else:
  print("No relevant posts found on Stack Overflow.")