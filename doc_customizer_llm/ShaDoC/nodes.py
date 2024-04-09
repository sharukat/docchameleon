import sys
import numpy as np
from lib.config import COLOR
from operator import itemgetter

from lib.common import GraphState, image
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


from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness,
    answer_similarity
)

with image.imports():
    import pandas as pd
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
        self.title: str = None
        self.question: str = None
        self.api_name: str = None
        self.issue_type: str = None
        self.iter = None
        self.context_iter = None
        self.intent: str = None
        self.so_answers = None
        self.course_urls = None
        self.ground_truth = None

        self.context = None
        self.definition = None
        self.task = None
        self.urls: list = None
        self.documentation = None
        self.debug = debug
        self.model ="gpt-4-0125-preview"
        self.node_map = {
            "intent_soanswers_courses": self.intent_soanswers_courses,
            "context_retrieval": self.context_retrieval,
            "check_context_relevancy": self.check_context_relevancy,
            "check_hallucination_and_answer_relevancy": self.check_hallucination_and_answer_relevancy,
            "check_issue_type": self.check_issue_type,
            "generate": self.generate,
            "generate_without_examples":self.generate_without_examples,
            "check_code_imports": self.check_code_imports,
            "check_code_execution": self.check_code_execution,
            "ragas_eval": self.ragas_eval,
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
    def intent_soanswers_courses(self, state: GraphState) -> GraphState:
        state_dict = state["keys"]
        self.title = state_dict["title"]
        self.question = state_dict["question"]
        self.api_name = state_dict["api_name"]
        self.issue_type = state_dict["issue_type"]
        self.ground_truth = state_dict["ground_truth"]
        iterations = state_dict["iterations"]
        context_iter = state_dict["context_iter"]

        self.intention = intent.question_intent_identifier(self.title, self.question)
        # self.so_answers = stackoverflow.retrieval(self.title, self.question)
        # search_results = search.course_urls_retriever(self.intention)
        # course_urls = search_results['urls']
        # if not course_urls:
        #     self.course_urls = utils.remove_broken_urls(course_urls)

        return {
            "keys": {
                "intent": self.intention,
                # "so_answers": self.so_answers,
                # "course_urls": self.course_urls,
                "iterations": iterations,
                "context_iter": context_iter,
                "issue_type": self.issue_type,
                "api_name": self.api_name,
                "documentation": self.documentation,
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
                "issue_type": self.issue_type,
                "api_name": self.api_name,
                "documentation": self.documentation,
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
            grade = score['binary_score']
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
                "issue_type": self.issue_type,
                "api_name": self.api_name,
                "documentation": self.documentation,
            }
        }



    # ========================================================================================================================
    def check_hallucination_and_answer_relevancy(self, state: GraphState) -> GraphState:
        print(f"{COLOR['BLUE']}🚀: EXECUTING GRADER: Hallucination and Answer Relevancy Checker.{COLOR['ENDC']}")
        state_dict = state["keys"]
        documents = state_dict["documents"]
        generation = state_dict["generation"]
        iterations = state_dict["iterations"]
        context_iter = state_dict["context_iter"]

        documents = "\n".join([d.page_content for d in documents])
        documents = Document(page_content=documents)

        hg = graders.hallucination_grader()
        score = hg.invoke({"documents": documents, "generation": generation})
        answer_grounded = score['binary_score']
        if answer_grounded == "no":
            print(f"\t{COLOR['RED']}--- ➡️ DECISION: LLM GENERATION IS NOT GROUNDED ---{COLOR['ENDC']}")
            grade = "no"
        else:
            print(f"\t{COLOR['GREEN']}--- ➡️ DECISION: LLM GENERATION IS GROUNDED ---{COLOR['ENDC']}")
            ag = graders.answer_grader()
            score = ag.invoke({"question": self.question,"generation": generation})
            answer_relevancy = score['binary_score']
            if answer_relevancy == "yes":
                print(f"\t{COLOR['GREEN']}--- ➡️ DECISION: LLM GENERATION RESOLVES THE QUESTION ---{COLOR['ENDC']}")
                grade = "yes"
            else:
                grade = "no"
                print(f"\t{COLOR['RED']}--- ➡️ DECISION: LLM GENERATION DOES NOT RESOLVES THE QUESTION. Re-TRY ---{COLOR['ENDC']}")
        print(f"{COLOR['GREEN']}✅: EXECUTION COMPLETED{COLOR['ENDC']}\n")
        
        return {
            "keys": {
                "documents": documents, 
                "generation": generation,
                "grade": grade,
                "iterations": iterations,
                "context_iter": context_iter,
                "issue_type": self.issue_type,
                "api_name": self.api_name,
                "documentation": self.documentation,
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
                "issue_type": self.issue_type,
                "api_name": self.api_name,
                "documentation": self.documentation,
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
                    "issue_type": self.issue_type,
                    "api_name": self.api_name,
                    "documentation": self.documentation,
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
                    "issue_type": self.issue_type,
                    "api_name": self.api_name,
                    "documentation": self.documentation,
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
                "issue_type": self.issue_type,
                "api_name": self.api_name,
                "documentation": self.documentation,
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
                "issue_type": self.issue_type,
                "api_name": self.api_name,
                "documentation": self.documentation,
            }
        }


    # ========================================================================================================================
    def ragas_eval(self, state: GraphState) -> GraphState:
        state_dict = state["keys"]
        contexts = state_dict["context"]
        answer = state_dict["generation"][0]
        code_solution = state_dict["generation"]
        error = state_dict["error"]
        prefix = state_dict["prefix"]
        imports = state_dict["imports"]
        iterations = state_dict["iterations"]
        code = state_dict["code"]

        metrics = [
            answer_relevancy,
            faithfulness,
            context_recall,
            context_precision,
            answer_correctness,
            answer_similarity
        ]

        print(f"{COLOR['GREEN']}--- RAGAS EVALUATION ---{COLOR['ENDC']}")

        dtypes = {
            "answer": str,
            "contexts": list,
            "ground_truth": str,
        }
        
        df = pd.DataFrame(data={}, columns=['question'])
        for col_name, dtype in dtypes.items():
            df[col_name] = np.empty(len(df), dtype=dtype)

        utils.update_cell(df, 'question', 0, self.question)
        utils.update_cell(df, 'answer', 0, str(answer))
        utils.update_cell(df, 'contexts', 0, [contexts])
        utils.update_cell(df, 'ground_truth', 0, self.ground_truth)

        response_dataset = Dataset.from_dict({
            "question" : df['question'],
            "answer" : df['answer'],
            "contexts" : df['contexts'],
            "ground_truth" : df['ground_truth'],
        })

        results = evaluate(response_dataset, metrics)
        results_df = results.to_pandas()
        
        print(f"{COLOR['BLUE']}RESULTS{COLOR['ENDC']}")
        print(f"{COLOR['BLUE']}{'-'*10}{COLOR['ENDC']}")
        print(f"\t{COLOR['BLUE']}➡️ Faithfulness        : {results_df['faithfulness'][0]} {COLOR['ENDC']}")
        print(f"\t{COLOR['BLUE']}➡️ Context Recall      : {results_df['context_recall'][0]} {COLOR['ENDC']}")
        print(f"\t{COLOR['BLUE']}➡️ Context Precision   : {results_df['context_precision'][0]} {COLOR['ENDC']}")
        print(f"\t{COLOR['BLUE']}➡️ Answer Correctness  : {results_df['answer_correctness'][0]} {COLOR['ENDC']}")
        print(f"\t{COLOR['BLUE']}➡️ Answer Similarity   : {results_df['answer_similarity'][0]} {COLOR['ENDC']}")
        print(f"\t{COLOR['BLUE']}➡️ Answer Relevancy    : {results_df['answer_relevancy'][0]} {COLOR['ENDC']}")

        return {
            "keys": {
                "generation": code_solution,
                "error": error,
                "prefix": prefix,
                "imports": imports,
                "context": contexts,
                "iterations": iterations,
                "code": code,
                "issue_type": self.issue_type,
                "api_name": self.api_name,
                "documentation": self.documentation,
                "faithfulness": results_df["faithfulness"][0],
                "context_recall": results_df["context_recall"][0],
                "context_precision": results_df["context_precision"][0],
                "answer_correctness": results_df["answer_correctness"][0],
                "answer_similarity": results_df["answer_similarity"][0],
                "answer_relevancy": results_df["answer_relevancy"][0],
            }
        }



    # ========================================================================================================================
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
        # urls = state_dict["urls"]
        documentation = state_dict["documentation"]
        code_solution = state_dict["generation"][0]
        prefix = code_solution.prefix
        imports = code_solution.imports
        code = code_solution.code

        output = output_template(api_name, prefix, imports, code)
        final = documentation + output

    else:
        documentation = state_dict["documentation"]
        solution = state_dict["generation"][0]
        final = documentation + str(solution)

    return final