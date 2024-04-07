import os
import sys
import modal
from lib.config import COLOR
from operator import itemgetter

from lib.common import GraphState, image, stub
import graders
import retrieval
import sandbox
import tasks
import intent
import search
import stackoverflow
import lib.utils as utils
from templates.template_1 import return_template
from templates.output_template import output_template

with image.imports():
    from langchain.schema import Document
    from langchain.output_parsers.openai_tools import PydanticToolsParser
    from langchain.prompts import PromptTemplate
    from langchain_core.pydantic_v1 import BaseModel, Field
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from langchain_openai import ChatOpenAI
    from langchain_community.chat_models import ChatCohere
    from langchain_community.retrievers import CohereRagRetriever


class Nodes:
    def __init__(self, debug: bool = False):
        self.title = None
        self.question = None
        self.api_name = None
        self.issue_type = None
        self.iter = None
        self.context_iter = None
        self.intent: str = None
        self.so_answers = None
        self.course_urls = None

        self.context = None
        self.definition = None
        self.task = None
        self.urls: list = None
        self.documentation = None
        self.debug = debug
        self.model ="gpt-4-0125-preview"
        self.node_map = {
            "identify_intent": self.identify_intent,
            "extract_so": self.extract_so,
            "retrieve_courses": self.retrieve_courses,
            "context_retrieval": self.context_retrieval,
            "check_context_relevancy": self.check_context_relevancy,
            "check_hallucination": self.check_hallucination,
            "check_answer_relevancy": self.check_answer_relevancy,
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
    


    # ========================================================================================================================
    def identify_intent(self, state: GraphState) -> GraphState:
        state_dict = state["keys"]
        self.title = state_dict["title"]
        self.question = state_dict["question"]
        self.api_name = state_dict["api_name"]
        self.issue_type = state_dict["issue_type"]
        iterations = state_dict["iterations"]
        context_iter = state_dict["context_iter"]

        self.intention = intent.question_intent_identifier(self.title, self.question)

        return {
            "keys": {
                "intent": self.intention,
                "iterations": iterations,
                "context_iter": context_iter,
            }
        }
    

    # ========================================================================================================================
    def extract_so(self, state: GraphState) -> GraphState:
        state_dict = state["keys"]
        iterations = state_dict["iterations"]
        context_iter = state_dict["context_iter"]

        self.so_answers = stackoverflow.retrieval(self.title, self.question)

        return {
            "keys": {
                "so_answers": self.so_answers,
                "iterations": iterations,
                "context_iter": context_iter,
            }
        }


    # ========================================================================================================================
    def retrieve_courses(self, state: GraphState) -> GraphState:
        state_dict = state["keys"]
        iterations = state_dict["iterations"]
        context_iter = state_dict["context_iter"]

        search_results = search.course_urls_retriever(self.intention)
        course_urls = search_results['urls']
        if not course_urls:
            self.course_urls = utils.remove_broken_urls(course_urls)

        return {
            "keys": {
                "course_urls": self.course_urls,
                "iterations": iterations,
                "context_iter": context_iter,
            }
        }
    

    # ========================================================================================================================
    def context_retrieval(self, state: GraphState) -> GraphState:
        """
        Retrieve relevant context from web using Cohere RAG Retriever
        Args:
            state (dict): The current graph state
        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """

        state_dict = state["keys"]
        iterations = state_dict["iterations"]
        context_iter = state_dict["context_iter"]

        print(f"{COLOR['BLUE']}🚀: EXECUTING RETRIEVER: Cohere RAG Retriever{COLOR['ENDC']}")

        # RAG generation
        rag = CohereRagRetriever(llm=ChatCohere(model="command-r"), connectors=[{"id": "web-search"}])
        documents = rag.get_relevant_documents(self.intention)
        generation = documents.pop()
        generation = generation.page_content
        print(f"{COLOR['GREEN']}✅: EXECUTION COMPLETED{COLOR['ENDC']}\n")

        context_iter += 1
        return {
            "keys": {
                "documents": documents, 
                "generation": generation,
                "iterations": iterations,
                "context_iter": context_iter,
            }
        }
    

    # ========================================================================================================================
    def check_context_relevancy(self, state: GraphState) -> GraphState:
        """
        Determines whether the retrieved documents are relevant to the question.
        Args:
            state (dict): The current graph state
        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """
        print(f"{COLOR['BLUE']}🚀: EXECUTING GRADER: Context vs Question Checker.{COLOR['ENDC']}")

        state_dict = state["keys"]
        documents = state_dict["documents"]
        generation = state_dict["generation"]
        iterations = state_dict["iterations"]
        context_iter = state_dict["context_iter"]

        # Score each doc
        filtered_docs = []
        for d in documents:
            rg = graders.retrieval_grader()
            score = rg.invoke({"question": self.question, "document": d.page_content})
            grade = score.binary_score
            if grade == "yes":
                print(f"\t{COLOR['GREEN']}--- ➡️ GRADE: DOCUMENT RELEVANT ---{COLOR['ENDC']}")
                filtered_docs.append(d)
            else:
                print(f"\t{COLOR['RED']}--- ➡️ GRADE: DOCUMENT NOT RELEVANT ---{COLOR['ENDC']}")
                continue
        print(f"{COLOR['GREEN']}✅: EXECUTION COMPLETED{COLOR['ENDC']}\n")
        
        return {
            "keys": {
                "documents": filtered_docs, 
                "generation": generation,
                "iterations": iterations,
                "context_iter": context_iter,
            }
        }



    # ========================================================================================================================
    def check_hallucination(self, state: GraphState) -> GraphState:
        print(f"{COLOR['BLUE']}🚀: EXECUTING GRADER: Hallucination Checker.{COLOR['ENDC']}")
        state_dict = state["keys"]
        documents = state_dict["documents"]
        generation = state_dict["generation"]
        iterations = state_dict["iterations"]
        context_iter = state_dict["context_iter"]

        documents = "\n".join([d.page_content for d in documents])
        documents = Document(page_content=documents)

        hg = graders.hallucination_grader()
        score = hg.invoke({"documents": documents, "generation": generation})
        grade = score.binary_score

        print(f"{COLOR['GREEN']}✅: EXECUTION COMPLETED{COLOR['ENDC']}\n")
        
        return {
            "keys": {
                "documents": documents, 
                "generation": generation,
                "hallucinations": grade,
                "iterations": iterations,
                "context_iter": context_iter,
            }
        }
    


    # ========================================================================================================================
    def check_answer_relevancy(self, state: GraphState) -> GraphState:
        print(f"{COLOR['BLUE']}🚀: EXECUTING GRADER: Generation vs Question Checker.{COLOR['ENDC']}")
        state_dict = state["keys"]
        documents = state_dict["documents"]
        generation = state_dict["generation"]
        hallucinations = state_dict["hallucinations"]
        iterations = state_dict["iterations"]
        context_iter = state_dict["context_iter"]

        ag = graders.answer_grader()
        score = ag.invoke({"question": self.question,"generation": generation})
        grade = score.binary_score
        print(f"{COLOR['GREEN']}✅: EXECUTION COMPLETED{COLOR['ENDC']}\n")

        return {
            "keys": {
                "documents": documents, 
                "generation": generation,
                "hallucinations": hallucinations,
                "answer_relevancy": grade, 
                "iterations": iterations,
                "context_iter": context_iter,
            }
        }


    # ========================================================================================================================
    def check_issue_type(self, state: GraphState) -> GraphState:
        state_dict = state["keys"]
        context = state_dict["generation"]
        iterations = state_dict["iterations"]
        flag = False

        if self.issue_type in self.examples_required:
            flag = True

        return {
            "keys": {
                "context": context,
                "iterations": iterations,
                "example_required": flag,
                
            }
        }



    # ========================================================================================================================
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
        context = state_dict["context"]
        iterations = state_dict["iterations"]

        try:
            self.documentation = utils.get_documentation(self.api_name)
            results = tasks.prompt_task(self.issue_type)
            self.definition = results['definition']
            self.task = results['task']

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
                    input_variables=["context", "api_name", "title", "question", "documentation", 
                                     "issue_type", "definition", "task", "generation", "error"],
                )

                # Chain
                chain = (
                    {
                        "context": itemgetter("context"),
                        "documentation": lambda _: self.documentation,
                        "title": lambda _: self.title,
                        "question": lambda _: self.question,
                        "api_name": lambda _: self.api_name,
                        "issue_type": lambda _: self.issue_type,
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
                    {   "context": context,
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
                        "context": itemgetter("context"),
                        "documentation": lambda _: self.documentation,
                        "title": lambda _: self.title,
                        "question": lambda _: self.question,
                        "api_name": lambda _: self.api_name,
                        "issue_type": lambda _: self.issue_type,
                        "definition": lambda _: self.definition,
                        "task": lambda _: self.task,
                    }
                    | prompt
                    | llm_with_tool
                    | parser_tool
                )

                code_solution = chain.invoke(
                    {
                        "context": context,
                    }
                )
                

            iterations += 1
            return {
                "keys": {
                    "generation": code_solution,
                    "context": context,
                    "iterations": iterations,
                }
            }
    
        except ValueError as e:
            print(e)



    # ========================================================================================================================
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
        context = state_dict["context"]
        iterations = state_dict["iterations"]

        try:
            utils.is_valid_api(self.api_name)
            self.documentation = utils.get_documentation(self.api_name)

            results = tasks.prompt_task(self.issue_type)
            self.definition = results['definition']
            self.task = results['task']

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
                    "context": itemgetter("context"),
                    "documentation": lambda _: self.documentation,
                    "title": lambda _: self.title,
                    "question": lambda _: self.question,
                    "api_name": lambda _: self.api_name,
                    "issue_type": lambda _: self.issue_type,
                    "definition": lambda _: self.definition,
                    "task": lambda _: self.task,
                }
                | prompt
            )

            solution = chain.invoke(
                {
                    "context": context,
                }
            )
            
            return {
                "keys": {
                    "generation": solution,
                    "context": context,
                    "iterations": iterations,
                }
            }
    
        except ValueError as e:
            print(e)



    # ========================================================================================================================
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
        context = state_dict["context"]
        code_solution = state_dict["generation"]
        imports = code_solution[0].imports
        iterations = state_dict["iterations"]

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
                "error": error,
                "context": context,
                "iterations": iterations,
            }
        }



    # ========================================================================================================================
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
        context = state_dict["context"]
        code_solution = state_dict["generation"]
        prefix = code_solution[0].prefix
        imports = code_solution[0].imports
        code = code_solution[0].code
        code_block = imports + "\n" + code
        iterations = state_dict["iterations"]

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
                "error": error,
                "prefix": prefix,
                "imports": imports,
                "context": context,
                "iterations": iterations,
                "code": code,
            }
        }



    # ========================================================================================================================
    def finish(self, state: GraphState) -> dict:
        """
        Finish the graph

        Returns:
            dict: Final result
        """
        examples_required = [
            "Documentation Replication on Other Examples", 
            "Documentation Replicability", ]
        
        state_dict = state["keys"]
        if self.issue_type in examples_required:
            code_solution = state_dict["generation"][0]
            prefix = code_solution.prefix
            imports = code_solution.imports
            code = code_solution.code

            output = output_template(self.api_name, prefix, imports, code)
            final = self.documentation + output

        else:
            solution = state_dict["generation"][0]
            final = self.documentation + solution

        print(f"\n{COLOR['YELLOW']}🏁 FINISHING {COLOR['ENDC']}")

        # response = extract_response(state)

        return {"keys": {"response": final}}


# def extract_response(state: GraphState) -> str:
#     """
#     Extract the response from the graph state

#     Args:
#         state (dict): The current graph state

#     Returns:
#         str: The response
#     """
#     examples_required = [
#             "Documentation Replication on Other Examples", 
#             "Documentation Replicability", ]

#     state_dict = state["keys"]
#     issue_type = state_dict["issue_type"]
#     if issue_type in examples_required:
#         api_name = state_dict["api_name"]
#         # urls = state_dict["urls"]
#         documentation = state_dict["documentation"]
#         code_solution = state_dict["generation"][0]
#         prefix = code_solution.prefix
#         imports = code_solution.imports
#         code = code_solution.code

#         output = output_template(api_name, prefix, imports, code)
#         final = documentation + output

#     else:
#         documentation = state_dict["documentation"]
#         solution = state_dict["generation"][0]
#         final = documentation + solution

#     return final