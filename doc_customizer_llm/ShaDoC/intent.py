# ==========================================================================================================================
# IMPORT DEPENDENCIES
# ==========================================================================================================================

import os
import ast
from lib.config import COLOR
from dotenv import load_dotenv

# LANGCHAIN MODULES
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_core.output_parsers import JsonOutputParser


# ==========================================================================================================================
# MAIN CODE
# ==========================================================================================================================
class Output(BaseModel):
    intent: str = Field(description="intent or goal behind the question")

def question_intent_identifier(title, body):
    print(f"{COLOR['BLUE']}🚀: EXECUTING INTENT IDENTIFIER: Identifying intent of the SO question{COLOR['ENDC']}")

    parser = JsonOutputParser(pydantic_object=Output)
    format_instructions = parser.get_format_instructions()

    template = """
        Your task is to identify the intent behind questions about the TensorFlow API documentation.

        First, read the question title:
        {title}

        Then read the full question body:
        {body}

        After reading the title and body, identify the intent or goal behind the question - what is the user ultimately trying to 
        accomplish or understand?

        Do not make any external assumptions beyond the information provided in the title and body. Base the identified intent solely on 
        the given question text.

        {format_instructions}
    """

    PROMPT = PromptTemplate(
        template=template,
        input_variables=['title','body'], 
        partial_variables={"format_instructions":format_instructions})
    
    llm = ChatOpenAI(temperature=0, model_name="gpt-4-0125-preview")
    chain = PROMPT | llm | parser

    prompt = {"title" : title, "body" : body}
    response = chain.invoke(prompt)
    print(f"{COLOR['GREEN']}✅: EXECUTION COMPLETED{COLOR['ENDC']}\n")
    return response['intent']