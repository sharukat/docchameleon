# ==========================================================================================================================
# IMPORT DEPENDENCIES
# ==========================================================================================================================

from dotenv import load_dotenv
import requests
from langchain_community.document_loaders import YoutubeLoader
from langchain_cohere import CohereRagRetriever
from langchain_cohere import ChatCohere
from urllib.parse import urlparse
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

import pandas as pd
import re
import time
import os
load_dotenv(dotenv_path="/Users/sharukat/Documents/ResearchYU/Code/doc-customizer-llm/doc_customizer_llm/.env")
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")   # Claude LLM API Key
os.environ["VOYAGE_API_KEY"] = os.getenv("OPENAI_API_KEY") 

# ==========================================================================================================================
# MAIN CODE
# ==========================================================================================================================
# class Output(BaseModel):
#     urls: List[str] = Field(description="list of relevant online courses urls")
    

sources = ["https://www.edx.org/", "https://www.coursera.org/", "https://www.udemy.com/", "https://www.udacity.com/", "https://www.youtube.com/"]
source_path_checks = {
    "https://www.edx.org/": "&product_category=course&",
    "https://www.coursera.org/": "learn",
    "https://www.udemy.com/": "course",
    "https://www.udacity.com/": "course",
}

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

def is_url_accessible(url, timeout=5):
    try:
        response = requests.head(url, timeout=timeout)
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"Error accessing URL {url}: {e}")
        return False
    
phrase = "Not the answer you\'re looking for?"



def process(content, knowledge):
    answers = []
    embedding_function = CohereEmbeddings(model="embed-english-v3.0")
    text_splitter = SemanticChunker(embeddings=embedding_function, breakpoint_threshold_type="percentile")
    
    splits = text_splitter.split_documents(content)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_function)
    full_answer_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 3})

    compressor = CohereRerank(model='rerank-english-v3.0', top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=full_answer_retriever , 
    )
    
    reranked_answers = compression_retriever.invoke(input=knowledge)

    return reranked_answers if reranked_answers else None



def course_urls_retriever(query:str, knowledge: str, sources: list):
    print(f"🚀: EXECUTING: Background Knowledge Source Identifier")
    urls = set()
    docs = []
    answers = []

    for source in sources:
        rag = CohereRagRetriever(
            llm=ChatCohere(model="command-r-plus"), 
            connectors=[{"id": "web-search", "options": {"site": source}}])
        
        try:
            documents = rag.invoke(query)
            answers.append(documents[-1].page_content)

            for doc in documents:
                url = doc.metadata.get('url')
                if url and is_url_accessible(url):
                    parsed_url = urlparse(url)
                    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}/"
                    if base_url in source_path_checks:
                        path_segments = parsed_url.path.strip('/').split('/')
                        if path_segments and path_segments[0] == source_path_checks.get(source):
                            if url not in urls and len(urls) < 6:
                                urls.add(url)
                            

                    elif base_url == "https://www.youtube.com/":
                        try:
                            loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
                            content = loader.load()
                            reranked_answers = process(content, knowledge)
                            if reranked_answers is not None:
                                for index, answer in enumerate(reranked_answers):
                                    print(answer.metadata['relevance_score'])
                                    if answer.metadata['relevance_score'] > 0.5:
                                        if url not in urls and len(urls) < 6:
                                            urls.add(url)
                                            transcript = f'"""{content}"""'
                                            docs.append(transcript)
                            
                            else:
                                print("No relevant videos")
                                pass

                        except Exception as e:
                            print(e)
                            continue
                        
                    elif base_url == "https://stackoverflow.com/":
                        print("executing executing executing")
                        page_content = doc.page_content
                        page_content = page_content.split(phrase)[0]

                        for text in to_remove:
                            page_content = page_content.replace(text, "")

                        for pattern in patterns:
                            page_content = re.sub(pattern, '', page_content)

                        df = pd.DataFrame([{"document": page_content}])
                        loader = DataFrameLoader(df, page_content_column="document")
                        content = loader.load()

                        reranked_answers = process(content, knowledge)
                        if reranked_answers is not None:
                            for index, answer in enumerate(reranked_answers):
                                print(answer.metadata['relevance_score'])
                                if answer.metadata['relevance_score'] > 0.5:
                                    if url not in urls and len(urls) < 6:
                                        urls.add(url)
                                    answer_content = answer.page_content
                                    answer_content = f'"""{answer_content}"""'
                                    docs.append(answer_content)
                        else:
                            pass
                    time.sleep(8)
                else:
                    print("URL is not accessible")
                    pass
                            
        except Exception as e:
            print(e)
            continue

    print(answers)
    print(f"✅: EXECUTION COMPLETED\n")
    return urls, docs

question = """
<p>When <code>src</code> has shape <code>[?]</code>, <code>tf.gather(src, tf.where(src != 0))</code> returns a tensor with shape <code>[?, 0]</code>. I'm not sure how a dimension can have size 0, and I'm especially unsure how to change the tensor back. I didn't find anything in the documentation to explain this, either.</p>

<p>I tried to <code>tf.transpose(tensor)[0]</code>, but the first dimension of the transposed tensor has size 0 and cannot be accessed! What's wrong?</p>
"""

knowledge = ['Understanding Tensor Shapes and Dimensions in TensorFlow', 'Tensor Manipulation and Operations in TensorFlow', 'Handling and Debugging Shape Mismatches in TensorFlow', 'Advanced TensorFlow Techniques for Data Processing', 'TensorFlow Documentation and Best Practices for Tensor Operations']
knowledge = "\n".join(knowledge)

# knowledge = '\n'.join(know)
query = f"""
Retrieve relevant online courses related to TensorFlow covering the following knowledge.
{knowledge}

Provide the course syllabus following its URL for each course you identified.
"""

new_sources = ["https://stackoverflow.com/"]

urls, docs = course_urls_retriever(query, knowledge, sources)
print(urls)
print(docs)







# python3 doc_customizer_llm/ShaDoC/knowledge_identifier.py
# {'knowledge': ['Understanding Tensor Shapes and Dimensions in TensorFlow', 'Tensor Manipulation and Operations in TensorFlow', 'Handling and Debugging Shape Mismatches in TensorFlow', 'Advanced TensorFlow Techniques for Data Processing', 'TensorFlow Documentation and Best Practices for Tensor Operations']}
# python3 doc_customizer_llm/ShaDoC/knowledge_identifier.py
# {'knowledge': ['Understanding TensorFlow tensor shapes and dimensions', 'Using tf.gather and tf.where functions in TensorFlow', 'Handling tensors with zero dimensions in TensorFlow', 'Common pitfalls and debugging techniques in TensorFlow', 'Tensor manipulation and transformation in TensorFlow']}
# python3 doc_customizer_llm/ShaDoC/knowledge_identifier.py
# {'knowledge': ['Understanding the behavior of tf.gather and tf.where when dealing with tensors that have zero elements.', 'Explanation of how TensorFlow handles dimensions with size 0.', 'Methods to reshape or modify tensors that have dimensions with size 0.', 'Common pitfalls and solutions when working with TensorFlow operations that result in unexpected tensor shapes.', 'Techniques to debug and resolve shape mismatch issues in TensorFlow.']}

