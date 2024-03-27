import sys
import modal
from operator import itemgetter

import sandbox
from common import GraphState, image, stub
import retrieval
import intent_identifier

with image.imports():
    from langchain.output_parsers.openai_tools import PydanticToolsParser
    from langchain.prompts import PromptTemplate
    from langchain_core.pydantic_v1 import BaseModel, Field
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from langchain_openai import ChatOpenAI


@stub.cls(image=image, secrets=[modal.Secret.from_name("my-tf-secret")])
class Nodes:
    def __init__(self, debug: bool = False):
        self.intent = None
        self.context = None
        self.debug = debug
        self.model = (
            "gpt-4-0125-preview" if not self.debug else "gpt-3.5-turbo-0125"
        )
        self.node_map = {
            "generate": self.generate,
            "check_code_imports": self.check_code_imports,
            "check_code_execution": self.check_code_execution,
            "finish": self.finish,
        }

    def generate(self, state: GraphState) -> GraphState:
        """
        Generate a code solution based on LCEL docs and the input question
        with optional feedback from code execution tests

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """

        ## State
        state_dict = state["keys"]
        # title = state_dict["title"]
        question = state_dict["question"]
        iter = state_dict["iterations"]

        if self.intent is None and self.context is None:
            question_result = intent_identifier.question_intent(body=question)
            self.intent = question_result[0]['text']
            self.context = retrieval.relevant_context_retriever(query = self.intent)

        ## Data model
        class code(BaseModel):
            """Code output"""

            prefix: str = Field(
                description="Description of the problem and approach"
            )
            imports: str = Field(description="Code block import statements")
            code: str = Field(
                description="Code block not including import statements"
            )

        ## LLM
        llm = ChatOpenAI(temperature=0, model=self.model, streaming=True)

        # Tool
        code_tool_oai = convert_to_openai_tool(code)

        # LLM with tool and enforce invocation
        llm_with_tool = llm.bind(
            tools=[code_tool_oai],
            tool_choice={"type": "function", "function": {"name": "code"}},
        )

        # Parser
        parser_tool = PydanticToolsParser(tools=[code])

        ## Prompt
        template = (
            """You are a coding assistant with expertise in TensorFlow. \n
            You are able to execute Python TensorFlow code in a sandbox environment that was constructed by chaining together the following Dockerfile 
            commands: \n
            """
            + f"{image.dockerfile_commands()}"
            + """

            Your task is to provide a customized response to a user's question based on the given context. Here are the steps to follow:

            <Review Context>
            First, review the context provided in context. This context is important for understanding the background and scope of the task. 
            Only use this context and do not make any assumptions or rely on prior knowledge. Below is the context:
            \n ------- \n
            {context}
            \n ------- \n
            </Review Context>

            Your task is to generate complete executable code example with an explanation replicating the corresponding example in the documentation 
            to address the user's {question} by only using the knowledge provided as 'context'.

            Your response will be shown to the user.
            
            Answer the user question based on the above provided context. \n
            Ensure any code you provide can be executed with all required imports and variables defined. \n
            Structure your answer as a description of the code solution, \n
            then a list of the imports, and then finally list the functioning code block. \n
            Here is the user question again: \n --- --- --- \n {question}
        """
        )

        ## Generation
        if "error" in state_dict:
            print("---RE-GENERATE SOLUTION w/ ERROR FEEDBACK---")

            error = state_dict["error"]
            code_solution = state_dict["generation"]

            # Update prompt
            addendum = """  \n --- --- --- \n You previously tried to solve this problem. \n Here is your solution:
                        \n --- --- --- \n {generation}  \n --- --- --- \n  Here is the resulting error from code
                        execution:  \n --- --- --- \n {error}  \n --- --- --- \n Please re-try to answer this.
                        Structure your answer with a description of the code solution. \n Then list the imports.
                        And finally list the functioning code block. Structure your answer with a description of
                        the code solution. \n Then list the imports. And finally list the functioning code block.
                        \n Here is the user question: \n --- --- --- \n {question}"""
            template = template + addendum

            # Prompt
            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question", "generation", "error"],
            )

            # Chain
            chain = (
                {
                    "context": lambda _: self.context,
                    "question": itemgetter("question"),
                    "generation": itemgetter("generation"),
                    "error": itemgetter("error"),
                }
                | prompt
                | llm_with_tool
                | parser_tool
            )

            code_solution = chain.invoke(
                {
                    "question": question,
                    "generation": str(code_solution[0]),
                    "error": error,
                }
            )

        else:
            print("---GENERATE SOLUTION---")

            # Prompt
            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"],
            )

            # Chain
            chain = (
                {
                    "context": lambda _: self.context,
                    "question": itemgetter("question"),
                }
                | prompt
                | llm_with_tool
                | parser_tool
            )

            code_solution = chain.invoke({"question": question})

        iter = iter + 1
        return {
            "keys": {
                "generation": code_solution,
                "question": question,
                "iterations": iter,
            }
        }

    def check_code_imports(self, state: GraphState) -> GraphState:
        """
        Check imports

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, error
        """

        ## State
        print("---CHECKING CODE IMPORTS---")
        state_dict = state["keys"]
        question = state_dict["question"]
        code_solution = state_dict["generation"]
        imports = code_solution[0].imports
        iter = state_dict["iterations"]

        # Attempt to execute the imports
        sb = sandbox.run(imports)
        output, error = sb.stdout.read(), sb.stderr.read()
        if error:
            print("---CODE IMPORT CHECK: FAILED---")
            # Catch any error during execution (e.g., ImportError, SyntaxError)
            error = f"Execution error: {error}"
            print(f"Error: {error}", file=sys.stderr)
            if "error" in state_dict:
                error_prev_runs = state_dict["error"]
                error = (
                    error_prev_runs
                    + "\n --- Most recent run output and error --- \n"
                    " ------ output ------ \n"
                    + output
                    + "\n ------ error ------ \n"
                    + error
                )
        else:
            print("---CODE IMPORT CHECK: SUCCESS---")
            # No errors occurred
            error = "None"

        return {
            "keys": {
                "generation": code_solution,
                "question": question,
                "error": error,
                "iterations": iter,
            }
        }

    def check_code_execution(self, state: GraphState) -> GraphState:
        """
        Check code block execution

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, error
        """

        ## State
        print("---CHECKING CODE EXECUTION---")
        state_dict = state["keys"]
        question = state_dict["question"]
        code_solution = state_dict["generation"]
        prefix = code_solution[0].prefix
        imports = code_solution[0].imports
        code = code_solution[0].code
        code_block = imports + "\n" + code
        iter = state_dict["iterations"]

        sb = sandbox.run(code_block)
        output, error = sb.stdout.read(), sb.stderr.read()
        if error:
            print("---CODE BLOCK CHECK: FAILED---")
            error = f"Execution error: {error}"
            print(f"Error: {error}", file=sys.stderr)
            if "error" in state_dict:
                error_prev_runs = state_dict["error"]
                error = (
                    error_prev_runs
                    + "\n --- Most recent run output and error --- \n"
                    " ------ output ------ \n"
                    + output
                    + "\n ------ error ------ \n"
                    + error
                )
        else:
            print("---CODE BLOCK CHECK: SUCCESS---")
            # No errors occurred
            error = "None"

        return {
            "keys": {
                "generation": code_solution,
                "question": question,
                "error": error,
                "prefix": prefix,
                "imports": imports,
                "iterations": iter,
                "code": code,
            }
        }

    def finish(self, state: GraphState) -> dict:
        """
        Finish the graph

        Returns:
            dict: Final result
        """

        print("---FINISHING---")

        response = extract_response(state)

        return {"keys": {"response": response}}


def extract_response(state: GraphState) -> str:
    """
    Extract the response from the graph state

    Args:
        state (dict): The current graph state

    Returns:
        str: The response
    """

    state_dict = state["keys"]
    code_solution = state_dict["generation"][0]
    prefix = code_solution.prefix
    imports = code_solution.imports
    code = code_solution.code

    return "\n".join([prefix, imports, code])