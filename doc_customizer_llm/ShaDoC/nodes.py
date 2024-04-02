import os
import sys
import modal
from lib.config import COLOR, DOCUMENTATION_PATH
from operator import itemgetter

from lib.common import GraphState, image, stub
import sandbox
import retrieval
import tasks
import intent
import search
import stackoverflow
import lib.utils as utils
from templates.template_1 import return_template
from templates.output_template import output_template

with image.imports():
    from langchain.output_parsers.openai_tools import PydanticToolsParser
    from langchain.prompts import PromptTemplate
    from langchain_core.pydantic_v1 import BaseModel, Field
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from langchain_openai import ChatOpenAI


class Nodes:
    def __init__(self, debug: bool = False):
        self.intent: str = None
        self.context = None
        self.definition = None
        self.task = None
        self.course_urls = None
        self.urls: list = None
        self.so_answers = None
        self.documentation = None
        self.debug = debug
        self.model ="gpt-4-0125-preview"
        self.node_map = {
            "check_issue_type": self.check_issue_type,
            "generate": self.generate,
            "generate_without_examples":self.generate_without_examples,
            "check_code_imports": self.check_code_imports,
            "check_code_execution": self.check_code_execution,
            "finish": self.finish,
        }

        self.examples_required = [
            "Documentation Replication on Other Examples", 
            "Documentation Replicability", 
            "Inadequate Examples"]
        
        self.description_only = [
            "Documentation Ambiguity",
            "Documentation Completeness"]
    
    def check_issue_type(self, state: GraphState) -> GraphState:
        ## State
        state_dict = state["keys"]
        title = state_dict["title"]
        question = state_dict["question"]
        api_name = state_dict["api_name"]
        issue_type = state_dict["issue_type"]
        iter = state_dict["iterations"]
        flag = False

        if issue_type in self.examples_required:
            flag = True

        return {
            "keys": {
                "title": title,
                "question": question,
                "api_name": api_name,
                "issue_type": issue_type,
                "example_required": flag,
                "iterations": iter,
            }
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
        title = state_dict["title"]
        question = state_dict["question"]
        api_name = state_dict["api_name"]
        issue_type = state_dict["issue_type"]
        iter = state_dict["iterations"]

        try:
            utils.is_valid_api(api_name)
            self.documentation = utils.get_documentation(api_name)

            if self.intent is None:
                results = tasks.prompt_task(issue_type)
                self.definition = results['definition']
                self.task = results['task']

                self.intent = intent.question_intent_identifier(title, question)

                search_results = search.course_urls_retriever(self.intent)
                course_urls = search_results['urls']
                if not course_urls:
                    self.course_urls = utils.remove_broken_urls(course_urls)

                model_response, self.context, self.urls = retrieval.relevant_context_retriever(query = self.intent)

                self.so_answers = stackoverflow.retrieval(title, question)

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
            template = return_template(image)

            ## Generation
            if "error" in state_dict:
                print(f"{COLOR['RED']} {'='*10} 🔄 RE-GENERATE SOLUTION w/ ERROR FEEDBACK {'='*10} {COLOR['ENDC']}\n")

                error = state_dict["error"]
                code_solution = state_dict["generation"]

                # Update prompt
                addendum = """  
                        \n --- --- --- \n You previously tried to solve this problem. \n 
                        Here is your solution:
                        \n --- --- --- \n {generation}  \n --- --- --- \n  
                        
                        Here is the resulting error from code execution:  
                        \n --- --- --- \n {error}  \n --- --- --- \n 
                        Please re-try to answer this.
                        Structure your answer with a instructional description of the code solution. \n 
                        Then list the imports. And finally list the functioning code block. Structure your answer with a description of the code solution. \n 
                        Then list the imports. And finally list the functioning code block.
                        \n Here is the user question: \n --- --- --- \n {question}
                        """
                template = template + addendum

                # Prompt
                prompt = PromptTemplate(
                    template=template,
                    input_variables=["context", "api_name", "title","question", "documentation", 
                                     "issue_type", "definition", "task", "generation", "error"],
                )

                # Chain
                chain = (
                    {
                        "context": lambda _: self.context,
                        "documentation": lambda _: self.documentation,
                        "title": itemgetter("title"),
                        "question": itemgetter("question"),
                        "api_name": itemgetter("api_name"),
                        "issue_type": itemgetter("issue_type"),
                        "definition": lambda _: self.definition,
                        "task": lambda _: self.task,
                        "generation": itemgetter("generation"),
                        "error": itemgetter("error"),
                    }
                    | prompt
                    | llm_with_tool
                    | parser_tool
                )

                code_solution = chain.invoke(
                    {   
                        "title": title,
                        "question": question,
                        "api_name": api_name,
                        "issue_type": issue_type,
                        "generation": str(code_solution[0]),
                        "error": error,
                    }
                )

            else:
                print(f"{COLOR['YELLOW']} {'='*10} 🛠 GENERATE SOLUTION {'='*10} {COLOR['ENDC']}\n")

                # Prompt
                prompt = PromptTemplate(
                    template=template,
                    input_variables=['context', 'api_name', 'title', 'question', 'documentation',
                                    'issue_type', 'definition', 'task'])

                # Chain
                chain = (
                    {
                        "context": lambda _: self.context,
                        "documentation": lambda _: self.documentation,
                        "title": itemgetter("title"),
                        "question": itemgetter("question"),
                        "api_name": itemgetter("api_name"),
                        "issue_type": itemgetter("issue_type"),
                        "definition": lambda _: self.definition,
                        "task": lambda _: self.task,
                    }
                    | prompt
                    | llm_with_tool
                    | parser_tool
                )

                code_solution = chain.invoke(
                    {
                        "title": title,
                        "question": question,
                        "api_name": api_name,
                        "issue_type": issue_type,
                    }
                )
                

            iter = iter + 1
            return {
                "keys": {
                    "generation": code_solution,
                    "title": title,
                    "question": question,
                    "api_name": api_name,
                    "issue_type": issue_type,
                    "urls": self.urls,
                    "documentation": self.documentation,
                    "iterations": iter,
                }
            }
    
        except ValueError as e:
            print(e)


    def generate_without_examples(self, state: GraphState) -> GraphState:
        """
        Generate a description based on context and the input question

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """

        ## State
        state_dict = state["keys"]
        title = state_dict["title"]
        question = state_dict["question"]
        api_name = state_dict["api_name"]
        issue_type = state_dict["issue_type"]
        iter = state_dict["iterations"]

        try:
            utils.is_valid_api(api_name)
            self.documentation = utils.get_documentation(api_name)

            if self.intent is None:
                results = tasks.prompt_task(issue_type)
                self.definition = results['definition']
                self.task = results['task']

                self.intent = intent.question_intent_identifier(title, question)

                search_results = search.course_urls_retriever(self.intent)
                course_urls = search_results['urls']
                if not course_urls:
                    self.course_urls = utils.remove_broken_urls(course_urls)

                model_response, self.context, self.urls = retrieval.relevant_context_retriever(query = self.intent)

                self.so_answers = stackoverflow.retrieval(title, question)

            ## LLM
            llm = ChatOpenAI(temperature=0, model=self.model, streaming=True)

            ## Prompt
            template = return_template(image)


            print(f"{COLOR['YELLOW']} {'='*10} 🛠 GENERATE SOLUTION {'='*10} {COLOR['ENDC']}\n")

            # Prompt
            prompt = PromptTemplate(
                template=template,
                input_variables=['context', 'api_name', 'title', 'question', 'documentation',
                                'issue_type', 'definition', 'task'])

            # Chain
            chain = (
                {
                    "context": lambda _: self.context,
                    "documentation": lambda _: self.documentation,
                    "title": itemgetter("title"),
                    "question": itemgetter("question"),
                    "api_name": itemgetter("api_name"),
                    "issue_type": itemgetter("issue_type"),
                    "definition": lambda _: self.definition,
                    "task": lambda _: self.task,
                }
                | prompt
            )

            solution = chain.invoke(
                {
                    "title": title,
                    "question": question,
                    "api_name": api_name,
                    "issue_type": issue_type,
                }
            )
            
            return {
                "keys": {
                    "generation": solution,
                    "title": title,
                    "question": question,
                    "api_name": api_name,
                    "issue_type": issue_type,
                    "urls": self.urls,
                    "documentation": self.documentation,
                    "iterations": iter,
                }
            }
    
        except ValueError as e:
            print(e)



    def check_code_imports(self, state: GraphState) -> GraphState:
        """
        Check imports

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, error
        """

        ## State
        print(f"{COLOR['BLUE']}⏳ CHECKING CODE IMPORTS{COLOR['ENDC']}")
        print(f"{COLOR['BLUE']}{'-'*30}{COLOR['ENDC']}")
        state_dict = state["keys"]
        title = state_dict["title"]
        question = state_dict["question"]
        api_name = state_dict["api_name"]
        issue_type = state_dict["issue_type"]
        code_solution = state_dict["generation"]
        imports = code_solution[0].imports
        iter = state_dict["iterations"]

        # Attempt to execute the imports
        sb = sandbox.run(imports)
        output, error = sb.stdout.read(), sb.stderr.read()
        if error:
            print(f"\t{COLOR['RED']}--- ❌ CODE IMPORT CHECK: FAILED ---{COLOR['ENDC']}\n")
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
            print(f"\t{COLOR['GREEN']}--- ✅ CODE IMPORT CHECK: SUCCESS---{COLOR['ENDC']}\n")
            # No errors occurred
            error = "None"

        return {
            "keys": {
                "generation": code_solution,
                "title": title,
                "question": question,
                "api_name": api_name,
                "issue_type": issue_type,
                "urls": self.urls,
                "documentation": self.documentation,
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
        print(f"{COLOR['BLUE']}⏳ CHECKING CODE EXECUTION {COLOR['ENDC']}")
        print(f"{COLOR['BLUE']}{'-'*30}{COLOR['ENDC']}")
        state_dict = state["keys"]
        title = state_dict["title"]
        question = state_dict["question"]
        api_name = state_dict["api_name"]
        issue_type = state_dict["issue_type"]
        code_solution = state_dict["generation"]
        prefix = code_solution[0].prefix
        imports = code_solution[0].imports
        code = code_solution[0].code
        code_block = imports + "\n" + code
        iter = state_dict["iterations"]

        sb = sandbox.run(code_block)
        output, error = sb.stdout.read(), sb.stderr.read()
        if error:
            print(f"\t{COLOR['RED']}--- ❌ CODE BLOCK CHECK: FAILED ---{COLOR['ENDC']}\n")
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
            print(f"\t{COLOR['GREEN']}--- ✅ CODE BLOCK CHECK: SUCCESS ---{COLOR['ENDC']}\n")
            # No errors occurred
            error = "None"

        return {
            "keys": {
                "generation": code_solution,
                "title": title,
                "question": question,
                "api_name": api_name,
                "issue_type": issue_type,
                "urls": self.urls,
                "documentation": self.documentation,
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

        print(f"\n{COLOR['YELLOW']}🏁 FINISHING {COLOR['ENDC']}")

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
    examples_required = [
            "Documentation Replication on Other Examples", 
            "Documentation Replicability", ]

    state_dict = state["keys"]
    issue_type = state_dict["issue_type"]
    if issue_type in examples_required:
        api_name = state_dict["api_name"]
        urls = state_dict["urls"]
        documentation = state_dict["documentation"]
        code_solution = state_dict["generation"][0]
        prefix = code_solution.prefix
        imports = code_solution.imports
        code = code_solution.code

        output = output_template(api_name, prefix, imports, code, urls)
        final = documentation + output

    else:
        documentation = state_dict["documentation"]
        solution = state_dict["generation"][0]
        final = documentation + solution

    return final