from langchain_community.chat_models import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate

import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="/Users/sharukat/Documents/ResearchYU/Code/doc-customizer-llm/doc_customizer_llm/.env")
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")


def retrieval_grader():
    llm = ChatCohere(model="command-r", temperature=0)
    parser = JsonOutputParser(pydantic_object=GradeDocuments)

    template = """
        You are a grader assessing relevance of a retrieved document to a user question. \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.

        Retrieved document: \n\n {document} \n\n User question: {question}\n

        Strictly follow the format instructions given below.
        {format_instructions}
    """

    grade_prompt = PromptTemplate(
        template=template,
        input_variables=["document", "question"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    grader = grade_prompt | llm | parser
    return grader


question =  "What are the types of agent memory?"
document = """
    Agents are an emerging class of artificial intelligence (AI) systems that use large language models (LLMs) to 
    interact with the world. There are two main types of agent memory:\n- Short-term memory (STM)\n- Long-term memory (LTM)\n\n
    STM is the main context that is fed to the LLM in runtime. It is limited in size depending on the context window length 
    of the LLM. LTM, on the other hand, is the external context stored on disk. It is further divided into three 
    types:\n- Episodic memory (aka Raw memory)\n- Semantic memory (aka Reflections)\n- Procedural memory\n\nEpisodic 
    memory stores the ground truth of all the actions, their outputs, and the reasoning behind those actions. Semantic 
    memory stores an agent's knowledge about the world and itself. Procedural memory represents the agent's procedures 
    or thinking, acting, decision-making, etc
"""

rg = retrieval_grader()
score = rg.invoke({"question": question, "document": document})
grade = score['binary_score']
print(grade)
