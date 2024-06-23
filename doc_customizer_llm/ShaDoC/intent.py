# ==========================================================================================================================
# IMPORT DEPENDENCIES
# ==========================================================================================================================

from lib.config import COLOR

# LANGCHAIN MODULES
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_core.output_parsers import JsonOutputParser

import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="/Users/sharukat/Documents/ResearchYU/Code/doc-customizer-llm/doc_customizer_llm/.env")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")   # Claude LLM API Key


# ==========================================================================================================================
# MAIN CODE
# ==========================================================================================================================
class Output(BaseModel):
    intent: str = Field(description="intent or goal behind the question")
    keywords: list = Field(description="API names within the question body")

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
        accomplish or understand and emphasize the cause factor? And identify the API names within the question body.

        Do not make any external assumptions beyond the information provided in the title and body. Base the identified intent solely on 
        the given question text.

        {format_instructions}
    """

    PROMPT = PromptTemplate(
        template=template,
        input_variables=['title','body'], 
        partial_variables={"format_instructions":format_instructions})
    
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
    chain = PROMPT | llm | parser

    prompt = {"title" : title, "body" : body}
    response = chain.invoke(prompt)
    print(f"{COLOR['GREEN']}✅: EXECUTION COMPLETED{COLOR['ENDC']}\n")
    return response


# Test
# title = "Tensor has shape [?, 0] -- how to reshape to [?,]"
# question = """
# <p>When <code>src</code> has shape <code>[?]</code>, <code>tf.gather(src, tf.where(src != 0))</code> returns a tensor with shape <code>[?, 0]</code>. I'm not sure how a dimension can have size 0, and I'm especially unsure how to change the tensor back. I didn't find anything in the documentation to explain this, either.</p>

# <p>I tried to <code>tf.transpose(tensor)[0]</code>, but the first dimension of the transposed tensor has size 0 and cannot be accessed! What's wrong?</p>
# """

# response = question_intent_identifier(title, question)
# print(response)