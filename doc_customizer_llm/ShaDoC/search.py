# ==========================================================================================================================
# IMPORT DEPENDENCIES
# ==========================================================================================================================

import ast
from typing import List
from lib.config import COLOR
from dotenv import load_dotenv

# LANGCHAIN MODULES
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.agents import create_openai_tools_agent, load_tools
from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.tools.tavily_search import TavilySearchResults
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_core.output_parsers import JsonOutputParser



# ==========================================================================================================================
# MAIN CODE
# ==========================================================================================================================
class Output(BaseModel):
    urls: List[str] = Field(description="list of relevant online courses urls")
    

def course_urls_retriever(query: str):
    print(f"{COLOR['BLUE']}🚀: EXECUTING WEB SEARCH: Tavily Search API + GPT-4-Turbo{COLOR['ENDC']}")
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
        Output in only valid JSON format.

        And you should only use the following online course platforms to find relevant courses:

        - Coursera
        - Udemy
        - Edx
        - Udacity
 
        Do not use any other websites or online course platforms.

        Here is the input query again: \n --- --- --- \n {query}

        Strictly follow the below format instructions to format the output in json format.
        {format_instructions}

        Do not use any other text within the output. Do not use "```json" word in the output.
    """

    prompt = ChatPromptTemplate.from_messages([template])
    final_prompt = prompt.format_messages(query=query, format_instructions=format_instructions)
    res = agent_executor.invoke({"input": final_prompt})
    res =  ast.literal_eval(res['output'])
    print(f"{COLOR['GREEN']}✅: EXECUTION COMPLETED{COLOR['ENDC']}\n")
    return res
