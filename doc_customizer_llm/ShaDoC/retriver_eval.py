
from langchain_cohere import CohereRagRetriever
from langchain_cohere import ChatCohere
from urllib.parse import urlparse
import requests
import pandas as pd
import re
import time

import os
from dotenv import load_dotenv
# from lib.config import COLOR

from ragas.metrics import context_relevancy
from ragas import evaluate
from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DataFrameLoader
from langchain_cohere import CohereEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Chroma
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

load_dotenv(dotenv_path="/Users/sharukat/Documents/ResearchYU/Code/doc-customizer-llm/doc_customizer_llm/.env")
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")   # Claude LLM API Key
# os.environ["VOYAGE_API_KEY"] = os.getenv("VOYAGE_API_KEY") 

def is_url_accessible(url, timeout=5):
    try:
        response = requests.head(url, timeout=timeout)
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"Error accessing URL {url}: {e}")
        return False

# 

def context_retrieval(question, isMaxContext=False):
        print(f"🚀: EXECUTING RETRIEVER: Cohere RAG Retriever")
        
        max_score = 0
        max_contexts = []
        all_contexts = []
        llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
        so_urls = set()
        docs = []
        scores = []

        embedding_function = CohereEmbeddings(model="embed-english-v3.0")
        text_splitter = SemanticChunker(embeddings=embedding_function, breakpoint_threshold_type="percentile")
        
        # for source in sources:
        to_remove = {
            "Stack Overflow Public questions & answers\n\n",
            "Stack Overflow for Teams Where developers & technologists share private knowledge with coworkers\n\n",
            "Find centralized, trusted content and collaborate around the technologies you use most. Learn more about Collectives\n\n",
            "Connect and share knowledge within a single location that is structured and easy to search. Learn more about Teams\n\n",
            "\n\nTo subscribe to this RSS feed, copy and paste this URL into your RSS reader.",
            "What makes a homepage useful for logged-in users\n\n",
            "Skip to main content\n\n",
            "Stack Overflow Public questions & answers\n\n",
            "Stack Overflow for Teams Where developers & technologists share private knowledge with coworkers\n\n",
            "Talent Build your employer brand\n\n",
            "Advertising Reach developers & technologists worldwide\n\n",
            "Labs The future of collective knowledge sharing\n\n",
            "Collectives™ on Stack Overflow\n\n",
            "Find centralized, trusted content and collaborate around the technologies you use most. Learn more about Collectives\n\n",
            "Connect and share knowledge within a single location that is structured and easy to search. Learn more about Teams\n\n",
            "Improve this question\n\n",
            "Sorted by: Reset to default\n\n",
            "Highest score (default)\n\n",
            "Trending (recent votes count more)\n\n",
            "Date modified (newest first)\n\n",
            "Date created (oldest first)\n\n",
            "Take the 2024 Developer Survey\n\n",
            "Get early access and see previews of new features. Learn more about Labs\n\n",
            "2024 Developer survey is here and we would like to hear from you!",
            "OverflowAI GenAI features for Teams\n"
            "New! OverflowAI: Where Community & AI Come Together\n"
        }
        
        patterns = {
             r'\n\n– \w+ [A-Za-z]{3} \d{1,2}, \d{4} at \d{1,2}:\d{2}', 
             r'\d{1,3}(?:,\d{3})*(?:\.\d+)?k?\d*\s*gold\s+badges?\d*\s*silver\s+badges?\d*\s*bronze\s+badges?\d*',
             r'answered\s+[A-Za-z]{3}\s+\d{1,2},\s+\d{4}\s+at\s+\d{1,2}:\d{2}\n\n',
             r'edited\s+[A-Za-z]{3}\s+\d{1,2},\s+\d{4}\s+at\s+\d{1,2}:\d{2}\n\n',
             r'asked\s+[A-Za-z]{3}\s+\d{1,2},\s+\d{4}\s+at\s+\d{1,2}:\d{2}\n\n',
             r'(\d+(?:\.\d+)?k?\d*\s*gold\s+badges?\s*)?(\d+(?:\.\d+)?k?\d*\s*silver\s+badges?\s*)?(\d+(?:\.\d+)?k?\d*\s*bronze\s+badges?\s*)?',
             r'.*?\s+Commented\s+[A-Za-z]{3}\s+\d{1,2},\s+\d{4}\s+at\s+\d{1,2}:\d{2}',
             }

        phrase = "Not the answer you\'re looking for?"

        try:
            rag = CohereRagRetriever(
                    llm=ChatCohere(model="command-r-plus"), 
                    connectors=[{"id": "web-search", "options": {"site": "https://stackoverflow.com/"}}])
            documents = rag.invoke(question)

            if documents:
                for doc in documents:
                    url = doc.metadata.get('url')
                    parsed_url = urlparse(url)
                    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}/"
                    if base_url ==  "https://stackoverflow.com/":
                        page_content = doc.page_content
                        page_content = page_content.split(phrase)[0]
                        # page_content = page_content.split("Learn more about Teams")[1]

                        for text in to_remove:
                            page_content = page_content.replace(text, "")

                        for pattern in patterns:
                            page_content = re.sub(pattern, '', page_content)

                        df = pd.DataFrame([{"document": page_content}])
                        loader = DataFrameLoader(df, page_content_column="document")
                        content = loader.load()

                        splits = text_splitter.split_documents(content)
                        vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_function)
                        full_answer_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 3})

                        compressor = CohereRerank(model='rerank-english-v3.0', top_n=3)
                        compression_retriever = ContextualCompressionRetriever(
                            base_compressor=compressor, 
                            base_retriever=full_answer_retriever , 
                        )

                        reranked_answers = compression_retriever.invoke(input=question)
                        if reranked_answers:
                            if url not in so_urls:
                                if is_url_accessible(url):
                                    so_urls.add(url)
                            for index, answer in enumerate(reranked_answers):
                                if answer.metadata['relevance_score'] > 0.5:
                                    answer_content = answer.page_content
                                    answer_content = f'"""{answer_content}"""'
                                    docs.append(answer_content)
                                        
                        time.sleep(8)
                    break
                    
            

                # if not documents:
                #      docs.append(["None"])
                #      continue
                # else:
                #     docs.append(documents)
                
                # generation = documents.pop()
                # contexts = generation.page_content
                # data = {"question": [question], "contexts": [[contexts]]}
                # results = evaluate(Dataset.from_dict(data), metrics=[context_relevancy], llm=llm)
                # res = results.to_pandas()
                # score = res['context_relevancy'][0]

                # all_contexts.append(contexts)
                # scores.append(score)

                # if source == "https://stackoverflow.com/":
                #     for doc in documents:
                #         url = doc.metadata.get('url')
                #         if url and is_url_accessible(url):
                #             so_urls.add(url)

                # if isMaxContext == True:
                #     if len(max_contexts) == 0:
                #          max_contexts.append({"context":contexts, "relevance_score":score})
                #          max_score = score
                #     else:
                #         if score > max_score:
                #             max_contexts.pop()
                #             max_contexts.append({"context":contexts, "relevance_score":score})
                #             max_score = score

            
                # if score > 0.5:
                #     print(f"\t ➡️ GRADE: DOCUMENT RELEVANT => Score: {score}")
                #     # if source == "https://stackoverflow.com/":
                #     #     for doc in documents:
                #     #         url = doc.metadata.get('url')
                #     #         if url and is_url_accessible(url):
                #     #             so_urls.add(url)

                #     if isMaxContext == True:
                        # if len(max_contexts) == 0:
                        #     max_contexts.append({"context":contexts, "relevance_score":score})
                        #     max_score = score
                        # else:
                        #     if score > max_score:
                        #         max_contexts.pop()
                        #         max_contexts.append({"context":contexts, "relevance_score":score})
                        #         max_score = score
                #     else:
                #         all_contexts.append(contexts)
                # else:
                #     print(f"\t ➡️ GRADE: DOCUMENT NOT RELEVANT")
                    
        except Exception as e:
            print(e)


        # if isMaxContext == False:
        #     if not all_contexts:
        #             all_contexts.append("N/A")
        # else:
        #     if not max_contexts:
        #             max_contexts.append("N/A")
        
        print(f"✅: EXECUTION COMPLETED\n")
        # return so_urls, max_contexts if isMaxContext else all_contexts
        # return docs, all_contexts, scores, so_urls
        return docs, so_urls


# question = "How to get ArrayList<Integer> and Scanner to play nice?"
# docs, urls = context_retrieval(question)
# print(docs)
# # print(urls)