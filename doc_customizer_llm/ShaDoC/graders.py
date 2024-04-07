from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.chat_models import ChatCohere
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

def retrieval_grader():

    # LLM with function call
    llm = ChatCohere(model="command-r", temperature=0)
    parser = PydanticOutputParser(pydantic_object=GradeDocuments)

    template = """
        You are a grader assessing relevance of a retrieved document to a user question. \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.

        Retrieved document: \n\n {document} \n\n User question: {question}\n

        {format_instructions}
    """

    grade_prompt = PromptTemplate(
        template=template,
        input_variables=["document", "question"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    grader = grade_prompt | llm | parser
    return grader



class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

def hallucination_grader():

    # LLM with function call
    llm = ChatCohere(model="command-r", temperature=0)
    parser = PydanticOutputParser(pydantic_object=GradeHallucinations)
    
    template = """
        You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
        Give a binary score 'yes' or 'no'. 'yes' means that the answer is grounded in / supported by the set of facts.\n
        'no' means that the answer is not grounded in / not supported by the set of facts.

        Set of facts: \n\n {documents} \n\n LLM generation: {generation}\n

        {format_instructions}
    """

    hallucination_prompt = PromptTemplate(
        template=template,
        input_variables=["documents", "generation"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )


    grader = hallucination_prompt | llm | parser
    return grader



class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

def answer_grader():

    # LLM with function call
    llm = ChatCohere(model="command-r", temperature=0)
    parser = PydanticOutputParser(pydantic_object=GradeAnswer)
    
    template = """
        You are a grader assessing whether an answer addresses / resolves a question \n
        Give a binary score 'yes' or 'no'. 'yes' means that the answer resolves the question.

        User question: \n\n {question} \n\n LLM generation: {generation}\n

        {format_instructions}
    """

    answer_prompt = PromptTemplate(
        template=template,
        input_variables=["question", "generation"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    grader = answer_prompt | llm | parser
    return grader