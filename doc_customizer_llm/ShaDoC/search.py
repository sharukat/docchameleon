# ==========================================================================================================================
# IMPORT DEPENDENCIES
# ==========================================================================================================================

# import ast
# from typing import List
from lib.config import COLOR
from dotenv import load_dotenv

# LANGCHAIN MODULES
# from langchain import hub
# from langchain_openai import ChatOpenAI
# from langchain.agents import AgentExecutor
# from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
# from langchain.agents import create_openai_tools_agent, load_tools
# # from langchain.utilities.tavily_search import TavilySearchAPIWrapper
# from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
# # from langchain.tools.tavily_search import TavilySearchResults
# from langchain_community.tools.tavily_search.tool import TavilySearchResults
# from langchain_core.pydantic_v1 import BaseModel, Field, validator
# from langchain_core.output_parsers import JsonOutputParser


from langchain_cohere import CohereRagRetriever
from langchain_cohere import ChatCohere
from urllib.parse import urlparse
from lib.utils import is_url_accessible

import time
import os
load_dotenv(dotenv_path="/Users/sharukat/Documents/ResearchYU/Code/doc-customizer-llm/doc_customizer_llm/.env")
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")   # Claude LLM API Key
# os.environ["VOYAGE_API_KEY"] = os.getenv("OPENAI_API_KEY") 

# ==========================================================================================================================
# MAIN CODE
# ==========================================================================================================================
# class Output(BaseModel):
#     urls: List[str] = Field(description="list of relevant online courses urls")
    

sources = ["https://www.edx.org/", "https://www.coursera.org/", "https://www.udemy.com/", "https://www.udacity.com/", "https://www.youtube.com/"]
source_path_checks = {
    "https://www.edx.org/": "learn",
    "https://www.coursera.org/": "learn",
    "https://www.udemy.com/": "course",
    "https://www.udacity.com/": "course",
    "https://www.youtube.com/": "video",
}



def course_urls_retriever(query: str):
    print(f"{COLOR['BLUE']}🚀: EXECUTING: Background Knowledge Source Identifier{COLOR['ENDC']}")
    # agent_llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
    # agent_prompt = hub.pull("hwchase17/openai-tools-agent")
    # search = TavilySearchAPIWrapper()
    # tavily_tool = TavilySearchResults(api_wrapper=search)
    # tools = [tavily_tool]
    # agent = create_openai_tools_agent(agent_llm, tools, agent_prompt)
    # agent_executor = AgentExecutor(agent=agent, tools=tools)

    # parser = JsonOutputParser(pydantic_object=Output)
    # format_instructions = parser.get_format_instructions()


    # template = """
    #     Your task is to identify relevant online courses based on the input {query}. 
    #     You have access to the search tools.
    #     Output in only valid JSON format.

    #     And you should only use the following online course platforms to find relevant courses:

    #     - Coursera
    #     - Udemy
    #     - Edx
    #     - Udacity
 
    #     Do not use any other websites or online course platforms.

    #     Here is the input query again: \n --- --- --- \n {query}

    #     Strictly follow the below format instructions to format the output in json format.
    #     {format_instructions}

    #     Do not use any other text within the output. Do not use "```json" word in the output.
    # """

    # prompt = ChatPromptTemplate.from_messages([template])
    # final_prompt = prompt.format_messages(query=query, format_instructions=format_instructions)
    # res = agent_executor.invoke({"input": final_prompt})
    # res =  ast.literal_eval(res['output'])
    urls = set()
    yt_urls = set()
    for source in sources:
        rag = CohereRagRetriever(
            llm=ChatCohere(model="command-r"), 
            connectors=[{"id": "web-search", "options": {"site": source}}])
        documents = rag.invoke(query)

        for doc in documents:
            url = doc.metadata.get('url')
            parsed_url = urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}/"
            if base_url in source_path_checks:
                if url and is_url_accessible(url):
                    if source == "https://www.youtube.com/":
                        yt_urls.add(url)
                    else:
                        path_segments = parsed_url.path.strip('/').split('/')
                        if path_segments and path_segments[0] == source_path_checks.get(source):
                            urls.add(url)


    print(f"{COLOR['GREEN']}✅: EXECUTION COMPLETED{COLOR['ENDC']}\n")
    return urls, yt_urls

intent = "The user is trying to understand why there is a difference in the output between manually calculating a dense layer operation and using the tf.keras.layers.Dense implementation, despite setting the kernel and bias to the same values."
result, youtube = course_urls_retriever(intent)
print(result)
print(youtube)