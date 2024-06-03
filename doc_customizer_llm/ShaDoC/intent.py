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
    
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
    chain = PROMPT | llm | parser

    prompt = {"title" : title, "body" : body}
    response = chain.invoke(prompt)
    print(f"{COLOR['GREEN']}✅: EXECUTION COMPLETED{COLOR['ENDC']}\n")
    return response['intent']


# Test
# title = "How to create output_signature for tensorflow.dataset.from_generator"
# question = """
# <p>I have a generator yielding data and labels <code>yield data, labels</code> where the data is
# an <code>numpy.ndarray</code> with variable rows and 500 columns of type <code>dtype=float32</code> and the labels are integers of <code>numpy.int64</code>.</p>
# <p>I'm trying to pass this data into TensorFlow from_generator function to create a TensorFlow dataset: <code>tf.data.Dataset.from_generator</code></p>
# <p>The <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator" rel="nofollow noreferrer">docs</a> say that the from_generator function needs a parameter <code>output_signature</code> as an input. But I'm having trouble understanding how to build this output_signature.</p>
# <p>How can I make the output_signature for the generator I described?</p>
# <p>Thank you!</p>
# <p>Edit:
# I used <code>tf.type_spec_from_value</code> to get this:</p>
# <pre><code>dataset = tf.data.Dataset.from_generator(
#    datagen_row,
#    output_signature=(
#       tf.TensorSpec(shape=(None, 512), dtype=tf.float32, name=None),
#       tf.TensorSpec(shape=(), dtype=tf.int64, name=None)
#    )
# )
# </code></pre>
# <p>But is it correct to use None when the number of rows is varying for the first data type?</p>
# """

# intent = question_intent_identifier(title, question)
# print(intent)