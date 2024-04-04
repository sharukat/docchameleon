# ==========================================================================================================================
# IMPORT DEPENDENCIES
# ==========================================================================================================================

import os
from dotenv import load_dotenv
from importlib.machinery import SourceFileLoader
# LANGCHAIN MODULES
from langchain_community.chat_models import ChatCohere
from langchain.retrievers import CohereRagRetriever
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# ==========================================================================================================================
# LOAD API KEYS FROM THE .env FILE
# ==========================================================================================================================

load_dotenv()

# os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")   # Cohere Command-R LLM API Key
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")   # LangChain LLM API Key (To use LangSmith)

# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGCHAIN_PROJECT"] = "langchain-websearch"

# ==========================================================================================================================
# MAIN CODE
# ==========================================================================================================================
llm = OpenAI(temperature=0.1)


def question_intent(query: str):
    prompt_template = PromptTemplate(input_variables=['query'],
                                     template='''
                                     Following is a question body. Understand the intent of this question.
                                     Simply give me a short modified question using the intent and also preserve the
                                     important API keywords. Do not add any extra words.
                                     {query}
                                     '''
                                     )

    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    return llm_chain.run(query)


def relevant_context_retriever(query: str):
    print("Retrieving relevant context from web using Cohere RAG Retriever....")
    urls = []
    page_content = []
    rag = CohereRagRetriever(llm=ChatCohere(model="command-r"),
                             connectors=[{"id": "web-search", 'options': {'site': 'https://www.youtube.com/'}}])
    docs = rag.get_relevant_documents(query)
    for doc in docs[:-1]:
        urls.append(doc.metadata['url'])
        page_content.append(doc.page_content)

    print("Completed Successfully\n")
    return urls, page_content
