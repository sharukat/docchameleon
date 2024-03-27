# ==========================================================================================================================
# IMPORT DEPENDENCIES
# ==========================================================================================================================

import os
from dotenv import load_dotenv

COLOR = {
    "HEADER": "\033[95m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "RED": "\033[91m",
    "ENDC": "\033[0m",
}

# LANGCHAIN MODULES
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic


# ==========================================================================================================================
# LOAD API KEYS FROM THE .env FILE
# ==========================================================================================================================

load_dotenv(dotenv_path="/Users/sharukat/Documents/ResearchYU/Code/doc-customizer-llm/doc_customizer_llm/.env")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")   # Claude LLM API Key
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")   # LangChain LLM API Key (To use LangSmith)

# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGCHAIN_PROJECT"] = "langchain-doc-customizer"


# ==========================================================================================================================
# MAIN CODE
# ==========================================================================================================================

def ai_doc_customizer(PROMPT, prompt):
    print(f"{COLOR['BLUE']}🚀: EXECUTING DOCUMENTATION CUSTOMIZER{COLOR['ENDC']}")
    llm = ChatOpenAI(temperature=0, model_name="gpt-4-0125-preview")
    chain = LLMChain(llm=llm, prompt=PROMPT)
    response = chain.apply(prompt)
    print(f"{COLOR['GREEN']}✅: CUSTOMIZATION COMPLETED{COLOR['ENDC']}\n")
    return response