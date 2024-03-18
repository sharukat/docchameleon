# ==========================================================================================================================
# IMPORT DEPENDENCIES
# ==========================================================================================================================

import os
from dotenv import load_dotenv

# LANGCHAIN MODULES
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic


# ==========================================================================================================================
# LOAD API KEYS FROM THE .env FILE
# ==========================================================================================================================

load_dotenv(dotenv_path="/Users/sharukat/Documents/ResearchYU/Code/doc-customizer-llm/doc_customizer_llm/.env")
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")   # Claude LLM API Key
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")   # LangChain LLM API Key (To use LangSmith)

# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGCHAIN_PROJECT"] = "langchain-doc-customizer"


# ==========================================================================================================================
# MAIN CODE
# ==========================================================================================================================

def ai_doc_customizer(PROMPT, prompt):
    print("Final API documentation customization process initiated....")
    llm = ChatAnthropic(temperature=0, model_name="claude-3-opus-20240229")
    synopsis_chain = LLMChain(llm=llm, prompt=PROMPT)
    response = synopsis_chain.apply(prompt)
    print("Customization completed !")
    return response