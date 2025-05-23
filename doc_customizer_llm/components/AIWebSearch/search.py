# ==========================================================================================================================
# IMPORT DEPENDENCIES
# ==========================================================================================================================

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.tools.tavily_search import TavilySearchResults
from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain import hub
import os
import ast
from typing import List
from dotenv import load_dotenv

COLOR = {
    "HEADER": "\033[95m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "RED": "\033[91m",
    "ENDC": "\033[0m",
}


# ==========================================================================================================================
# LOAD API KEYS FROM THE .env FILE
# ==========================================================================================================================

load_dotenv(dotenv_path="PATRH TO THE ENV FILE")
os.environ["OPENAI_API_KEY"] = os.getenv(
    "OPENAI_API_KEY")  # Claude LLM API Key
os.environ["LANGCHAIN_API_KEY"] = os.getenv(
    "LANGCHAIN_API_KEY"
)  # LangChain LLM API Key (To use LangSmith)
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "langchain-websearch"


# ==========================================================================================================================
# MAIN CODE
# ==========================================================================================================================
class Output(BaseModel):
    urls: List[str] = Field(description="list of relevant online courses urls")


def course_urls_retriever(query: str):
    print(
        f"{COLOR['BLUE']}🚀: INITIATING WEB SEARCH: Tavily Search API + GPT-4-Turbo{COLOR['ENDC']}"
    )
    agent_llm = ChatOpenAI(temperature=0, model_name="gpt-4-0125-preview")
    agent_prompt = hub.pull("hwchase17/openai-tools-agent")
    search = TavilySearchAPIWrapper()
    tavily_tool = TavilySearchResults(api_wrapper=search)
    tools = [tavily_tool]
    agent = create_openai_tools_agent(agent_llm, tools, agent_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    parser = JsonOutputParser(pydantic_object=Output)
    format_instructions = parser.get_format_instructions()

    template = """
        Your task is to identify relevant online courses based on the input {query}. 
        You have access to the search tools.

        And you should only use the following online course platforms to find relevant courses:

        - Coursera
        - Udemy
        - Edx
        - Udacity
 
        Do not use any other websites or online course platforms.

        Here is the input query again: \n --- --- --- \n {query}

        {format_instructions}

        Do not use any other text within the output. Strictly follow the format instructions to format the output.
        Do not use "```json" word in the output.
    """

    prompt = ChatPromptTemplate.from_messages([template])
    final_prompt = prompt.format_messages(
        query=query, format_instructions=format_instructions
    )
    res = agent_executor.invoke({"input": final_prompt})
    res = ast.literal_eval(res["output"])
    print(f"{COLOR['GREEN']}✅: EXECUTION COMPLETED{COLOR['ENDC']}\n")
    return res
