# ==========================================================================================================================
# IMPORT DEPENDENCIES
# ==========================================================================================================================

import os
from dotenv import load_dotenv

# LANGCHAIN MODULES
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_openai_functions_agent
from langchain_community.tools.tavily_search import TavilySearchResults


# ==========================================================================================================================
# LOAD API KEYS FROM THE .env FILE
# ==========================================================================================================================

load_dotenv(dotenv_path="/Users/sharukat/Documents/ResearchYU/Code/doc-customizer-llm/doc_customizer_llm/.env")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")   # Claude LLM API Key
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")   # LangChain LLM API Key (To use LangSmith)

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "langchain-websearch"


# ==========================================================================================================================
# MAIN CODE
# ==========================================================================================================================

agent_llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)
# agent_prompt = "You are a helpful assistant and an expert in searching web"
agent_prompt = hub.pull("hwchase17/openai-functions-agent")
search = TavilySearchResults()
tools = [search]
agent = create_openai_functions_agent(agent_llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# ==========================================================================================================================
# TEST EXECUTIONS
# ==========================================================================================================================

query = "How to load a model using TensorFlow?"

system = """
    Identify the urls of highly relevant blogs and online courses on the below query. Ony use GeeksforGeeks or MachineLearningMastery
    websites to get data. And also, only use Coursera, Udemy, Edx or Udacity to identify courses that can be used to learn relevant
    concepts. Finally, return the response in markdown format.
    """
human = f"{query}"
input = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
res = agent_executor.invoke({"input": input})
print(res['output'])