# ==========================================================================================================================
# IMPORT DEPENDENCIES
# ==========================================================================================================================

# LANGCHAIN MODULES
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


# ==========================================================================================================================
# MAIN CODE
# ==========================================================================================================================

def question_intent(body: str):
    print("Identifying intent of the Stack Overflow question based on the question body......")

    template = """
        Your task is to identify the intent behind questions about the TensorFlow API documentation.

        Then read the full question body:
        {body}

        After reading the body, identify the intent or goal behind the question - what is the user ultimately trying to 
        accomplish or understand?

        Do not make any external assumptions beyond the information provided in the body. Base the identified intent solely on 
        the given question text.
    """

    PROMPT = PromptTemplate(
        template=template,
        input_variables=['body'])
    
    llm = ChatOpenAI(temperature=0, model_name="gpt-4-0125-preview")
    synopsis_chain = LLMChain(llm=llm, prompt=PROMPT)

    prompt = [{"body" : body},]
    response = synopsis_chain.apply(prompt)
    print("Completed Successfully\n")
    return response