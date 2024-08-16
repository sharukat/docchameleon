from ShaDoC.templates.template_2 import return_template
from ShaDoC.templates import template_1
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.utils.function_calling import convert_to_openai_tool

import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="/Users/sharukat/Documents/ResearchYU/Code/doc-customizer-llm/doc_customizer_llm/.env")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def generate_explanation(question, title=None, documentation=None, context = None, noContext = False):

    class Output(BaseModel):
        explanation: str = Field(description="generated explanation")

    parser = JsonOutputParser(pydantic_object=Output)
    format_instructions = parser.get_format_instructions()

    template = return_template(noContext)
    # print(f"{'='*10} 🛠 GENERATE SOLUTION {'='*10}\n")

    # if noContext == True:
    #     llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    #     task = "Your task is to generate an comprehensive explanation description only to address the question. Again, only explanation description only. No separete code examples should be included."

    #     prompt = PromptTemplate(
    #         template=template,
    #         input_variables=['title', 'question', 'task'])

    #     chain =  prompt | llm

    #     solution = chain.invoke({
    #         "title": title,
    #         "question": question,
    #         "task": task,
    #     })

    # else:
    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    task = "Your task is to generate an comprehensive explanation description only to address the question by only using the knowledge provided as 'context'. Again, only explanation description only. No separete code examples should be included."

    prompt = PromptTemplate(
    template=template,
    input_variables=['context', 'documentation', 'title', 'question', 'task'],
    )

    chain = prompt | llm

    solution = chain.invoke(
        {
            "context":context,
            "documentation": documentation,
            "title": title,
            "question": question,
            "task": task,
        }
    )

    # For Java quantitative analysis
    # llm = ChatOpenAI(temperature=0, model="gpt-4o")
    # task = "Your task is to generate an comprehensive explanation description only to address the question. Again, only explanation description only. No separete code examples should be included."

    # prompt = PromptTemplate(
    #     template=template,
    #     input_variables=['question', 'task'],
    #     partial_variables={"format_instructions":format_instructions}
    #     )

    # chain =  prompt | llm | parser

    # solution = chain.invoke({
    #     "question": question,
    #     "task": task,
    # })
    
    return solution




# Code generation for JAVA
# def generate_code(question):

#     class code(BaseModel):
#         """Code output"""
#         prefix: str = Field(description="Description of the problem and approach")
#         imports: str = Field(description="Code block import statements")
#         code: str = Field(description="Code block not including import statements")

#     llm = ChatOpenAI(temperature=0, model="gpt-4o")
#     code_tool_oai = convert_to_openai_tool(code)
#     llm_with_tool = llm.bind(
#         tools=[code_tool_oai],
#         tool_choice={"type": "function", "function": {"name": "code"}},
#     )
#     parser_tool = PydanticToolsParser(tools=[code])

#     template = template_1.return_template()

#     task = "Your task is to generate a executable code example to address the question."

#     prompt = PromptTemplate(
#         template=template,
#         input_variables=['question', 'task'],
#         )

#     chain =  prompt | llm_with_tool | parser_tool

#     solution = chain.invoke({
#         "question": question,
#         "task": task,
#     })
    
#     return solution