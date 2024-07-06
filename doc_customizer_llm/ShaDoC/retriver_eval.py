
from langchain_cohere import CohereRagRetriever
from langchain_cohere import ChatCohere
from urllib.parse import urlparse
import requests

import os
from dotenv import load_dotenv
from lib.config import COLOR

from ragas.metrics import context_relevancy
from ragas import evaluate
from datasets import Dataset
from langchain_openai import ChatOpenAI

load_dotenv(dotenv_path="/Users/sharukat/Documents/ResearchYU/Code/doc-customizer-llm/doc_customizer_llm/.env")
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")   # Claude LLM API Key
os.environ["VOYAGE_API_KEY"] = os.getenv("OPENAI_API_KEY") 

def is_url_accessible(url, timeout=5):
    try:
        response = requests.head(url, timeout=timeout)
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"Error accessing URL {url}: {e}")
        return False


def context_retrieval(intention, isMaxContext: bool = False):
        print(f"{COLOR['BLUE']}🚀: EXECUTING RETRIEVER: Cohere RAG Retriever{COLOR['ENDC']}")
        
        max_score = 0
        max_contexts = []
        all_contexts = []
        sources = ["https://stackoverflow.com/"]
        llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
        so_urls = set()
        
        for source in sources:
            try:
                rag = CohereRagRetriever(
                        llm=ChatCohere(model="command-r"), 
                        connectors=[{"id": "web-search", "options": {"site": source}}])
                documents = rag.invoke(intention)

                if not documents:
                     continue
                
                generation = documents.pop()
                contexts = generation.page_content
                data = {"question": [intention], "contexts": [[contexts]]}
                results = evaluate(Dataset.from_dict(data), metrics=[context_relevancy], llm=llm)
                res = results.to_pandas()
                score = res['context_relevancy'][0]
            
                if score > 0.5:
                    print(f"\t ➡️ GRADE: DOCUMENT RELEVANT => Score: {score}")
                    if source == "https://stackoverflow.com/":
                        for doc in documents:
                            url = doc.metadata.get('url')
                            if url and is_url_accessible(url):
                                so_urls.add(url)

                    if isMaxContext:
                        if score > max_score:
                            max_contexts = [contexts]
                            max_score = score
                    else:
                        all_contexts.append(contexts)
                else:
                    print(f"\t ➡️ GRADE: DOCUMENT NOT RELEVANT")
                    
            except Exception as e:
                 print(e)


        if not isMaxContext:
            if not all_contexts:
                    all_contexts.append("N/A")
        else:
            if not max_contexts:
                    max_contexts.append("N/A")
        
        print(f"{COLOR['GREEN']}✅: EXECUTION COMPLETED{COLOR['ENDC']}\n")
        return so_urls, max_contexts if isMaxContext else all_contexts


intent = "The user is trying to understand why there is a difference in the output between manually calculating a dense layer operation and using the tf.keras.layers.Dense implementation, despite setting the kernel and bias to the same values."
urls, result = context_retrieval(intent)
print(result)
print(urls)