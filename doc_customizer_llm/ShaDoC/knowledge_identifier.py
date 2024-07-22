
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_core.output_parsers import JsonOutputParser

import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="/Users/sharukat/Documents/ResearchYU/Code/doc-customizer-llm/doc_customizer_llm/.env")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def identify_knowledge(question, sources):
    class Output(BaseModel):
        nature: str = Field(description="Nature of the question")
        knowledge: list = Field(description="background knowledge that can be obtained from the given source or sources")

    parser = JsonOutputParser(pydantic_object=Output)
    format_instructions = parser.get_format_instructions()

    template = """
    You are an expert in identifying the knowledge that can be obtained from a given source based on a TensorFlow-related question. 

    First, think step-by-step and determine the nature of the question from the following categories: 
    - Data adaptation
    - Data cleaning
    - Error/Exception
    - Feature engineering
    - Loss function
    - Method selection
    - Model conversion
    - Model creation
    - Model load/store
    - Model reuse
    - Model selection
    - Model visualization
    - Optimizer
    - Output interpretation
    - Parameter selection
    - Performance
    - Prediction accuracy
    - Robustness
    - Setup
    - Shape mismatch

    Question:
    {question}

    Based on the identified nature of the question, list the knowledge that can be acquired through {sources} only. Provide your output in the format of a list.

    {format_instructions}

    """

    
    PROMPT = PromptTemplate(
        template=template,
        input_variables=['question', 'sources'], 
        partial_variables={"format_instructions":format_instructions})
    
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
    chain = PROMPT | llm | parser

    prompt = {"question" : question, "sources": sources}

    response = chain.invoke(prompt)
    return response


sources = "Stack Overflow Q&A"
question = """
<p>When <code>src</code> has shape <code>[?]</code>, <code>tf.gather(src, tf.where(src != 0))</code> returns a tensor with shape <code>[?, 0]</code>. I'm not sure how a dimension can have size 0, and I'm especially unsure how to change the tensor back. I didn't find anything in the documentation to explain this, either.</p>

<p>I tried to <code>tf.transpose(tensor)[0]</code>, but the first dimension of the transposed tensor has size 0 and cannot be accessed! What's wrong?</p>
"""

response = identify_knowledge(question)
print(response)