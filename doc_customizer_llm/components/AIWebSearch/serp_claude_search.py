# ==========================================================================================================================
# IMPORT DEPENDENCIES
# ==========================================================================================================================

import os
import ast
from dotenv import load_dotenv

# LANGCHAIN MODULES
from langchain import hub
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_openai_functions_agent, load_tools


# ==========================================================================================================================
# LOAD API KEYS FROM THE .env FILE
# ==========================================================================================================================

load_dotenv(dotenv_path="/Users/sharukat/Documents/ResearchYU/Code/doc-customizer-llm/doc_customizer_llm/.env")
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")   # Claude LLM API Key
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")   # LangChain LLM API Key (To use LangSmith)
os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "langchain-websearch"


# ==========================================================================================================================
# MAIN CODE
# ==========================================================================================================================

def course_urls_retriever(query: str):
    print("Seaching web for relevant online courses......")
    agent_llm = ChatAnthropic(temperature=0, model_name="claude-3-sonnet-20240229")
    agent_prompt = hub.pull("hwchase17/openai-functions-agent")
    tools = load_tools(["serpapi"])
    agent = create_openai_functions_agent(agent_llm, tools, agent_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


    system = """
        Your task is to identify relevant online courses based on the input query. 
        You have access to the search tools.

        And you should only use the following online course platforms to find relevant courses:

        - Coursera
        - Udemy
        - Edx
        - Udacity
 
        Do not use any other websites or online course platforms. Here's an example of how your JSON output should look:

        <example>
        {"course_urls": courses websites pages a s a list of URLs}
        </example>

        The output should only has the JSON output like the example shown above. Avoid adding any other text in to output response.

        Make sure to follow the instructions carefully and only use the allowed online course platforms. Return your output 
        in the specified JSON format.
    """

    human = f"Query: {query}"
    input = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    res = agent_executor.invoke({"input": input})
    res =  ast.literal_eval(res['output'])
    print("Completed Successfully\n")
    return res
