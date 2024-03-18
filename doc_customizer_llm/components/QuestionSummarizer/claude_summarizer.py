# ==========================================================================================================================
# IMPORT DEPENDENCIES
# ==========================================================================================================================

import os
from dotenv import load_dotenv

# LANGCHAIN MODULES
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic


# ==========================================================================================================================
# LOAD API KEYS FROM THE .env FILE
# ==========================================================================================================================

load_dotenv(dotenv_path="/Users/sharukat/Documents/ResearchYU/Code/doc-customizer-llm/doc_customizer_llm/.env")
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")   # Claude LLM API Key
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")   # LangChain LLM API Key (To use LangSmith)

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "langchain-question-summarizer"


# ==========================================================================================================================
# MAIN CODE
# ==========================================================================================================================

def question_summarizer(title, body):
    print("Identifying intent of the Stack Overflow question based on the question body......")

    template = """
        Your task is to identify the intent behind questions about the TensorFlow API documentation.
        And write the intention in two sentences.

        First, read the question title:
        {title}

        Then read the full question body:
        {body}

        After reading the title and body, identify the intent or goal behind the question - what is the user ultimately trying to 
        accomplish or understand?

        Do not make any external assumptions beyond the information provided in the title and body. Base the identified intent solely on 
        the given question text.
    """

    PROMPT = PromptTemplate(
        template=template,
        input_variables=['title','body'])
    
    llm = ChatAnthropic(temperature=0, model_name="claude-3-opus-20240229")
    synopsis_chain = LLMChain(llm=llm, prompt=PROMPT)

    prompt = [{"title" : title, "body" : body,},]
    response = synopsis_chain.apply(prompt)
    print("Completed Successfully\n")
    return response