from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
import os
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

load_dotenv(dotenv_path="../../.env")
os.environ["OPENAI_API_KEY"] = os.getenv(
    "OPENAI_API_KEY")  # Claude LLM API Key


def ai_doc_customizer(PROMPT, prompt):
    print(f"{COLOR['BLUE']}🚀: EXECUTING DOCUMENTATION CUSTOMIZER{COLOR['ENDC']}")
    llm = ChatOpenAI(temperature=0, model_name="gpt-4-0125-preview")
    chain = LLMChain(llm=llm, prompt=PROMPT)
    response = chain.apply(prompt)
    print(f"{COLOR['GREEN']}✅: CUSTOMIZATION COMPLETED{COLOR['ENDC']}\n")
    return response

