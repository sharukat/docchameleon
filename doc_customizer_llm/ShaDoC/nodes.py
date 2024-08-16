import sys
import ast
import json
import numpy as np
from datasets import Dataset
from operator import itemgetter

# Custom Modules
import time
import tasks
import knowledge_identifier
# import search
import sandbox
import graders
# import retrieval
# import stackoverflow
import lib.utils as utils
from lib.config import COLOR
from lib.common import GraphState, image
from templates import template_1, template_2, output_template


with image.imports():
    import pandas as pd
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        faithfulness,
        context_precision,
        context_relevancy,
        answer_similarity
    )
    # from setfit import SetFitModel
    from langchain.schema import Document
    from langchain_openai import ChatOpenAI
    from langchain.prompts import PromptTemplate
    # from langchain_community.chat_models import ChatCohere
    from langchain_cohere import ChatCohere
    from langchain_core.pydantic_v1 import BaseModel, Field
    # from langchain_community.retrievers import CohereRagRetriever
    from langchain_cohere import CohereRagRetriever
    from langchain.output_parsers.openai_tools import PydanticToolsParser
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from human_eval.execution import check_correctness
    from collections import defaultdict
    import nltk
    nltk.download('punkt')
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    
    
    
class Nodes:
    def __init__(self, debug: bool = False):
        self.problem = None
        self.input = None


        self.title: str = None
        self.question_id = None
        self.question: str = None
        self.api_name: str = None
        self.issue_type: str = None
        self.iter = None
        self.context_iter = None
        self.intent: str = None
        self.so_answers = None
        self.ground_truth = None

        self.context = None
        # self.definition = None
        # self.task = None
        self.urls: list = None
        self.documentation = None
        self.debug = debug
        self.model ="gpt-4o"
        self.node_map = {
            # "intent_soanswers_courses": self.intent_soanswers_courses,
            # "context_retrieval": self.context_retrieval,
            # "check_hallucination_and_answer_relevancy": self.check_hallucination_and_answer_relevancy,
            # "check_issue_type": self.check_issue_type,
            "generate": self.generate,
            # "generate_without_examples":self.generate_without_examples,
            "check_code_imports": self.check_code_imports,
            "check_code_execution": self.check_code_execution,
            # "ragas_eval": self.ragas_eval,
            "finish": self.finish,
        }
        self.sources = [
            "https://stackoverflow.com", 
        ]


    


    # ========================================================================================================================
    def intent_soanswers_courses(self, state: GraphState) -> GraphState:
        state_dict = state["keys"]
        self.title = state_dict["title"]
        self.question_id = state_dict['question_id']
        self.question = state_dict["question"]
        self.api_name = state_dict["api_name"]
        self.issue_type = state_dict["issue_type"]
        self.ground_truth = state_dict["ground_truth"]
        iterations = state_dict["iterations"]
        context_iter = state_dict["context_iter"]

        # Question type classification (multi-class classifier)
        # model = SetFitModel.from_pretrained("sharukat/sbert-issuetypeidentifier")
        # prediction = model(self.question)
        # if prediction == 0:
        #     self.issue_type = 'additional_resources'
        # elif prediction == 1:
        #     self.issue_type = 'examples_required'
        # else:
        #     self.issue_type = 'description_only'


        results = knowledge_identifier.generate_queries(self.question)
        self.intention = results['query']
        

        return {
            "keys": {
                "iterations": iterations,
                "context_iter": context_iter,
                "documentation": self.documentation,
            }
        }
    
    
    

    # ========================================================================================================================
    def context_retrieval(self, state: GraphState) -> GraphState:
        state_dict = state["keys"]
        iterations = state_dict["iterations"]
        context_iter = state_dict["context_iter"]

        print(f"{COLOR['BLUE']}🚀: EXECUTING RETRIEVER: Cohere RAG Retriever{COLOR['ENDC']}")

        documents_set = []
        generated_context = []

        # RAG generation
        for source in self.sources:
            rag = CohereRagRetriever(llm=ChatCohere(model="command-r"), connectors=[{"id": "web-search", "options": {"site": source}}])
            documents = rag.invoke(self.intention)
            time.sleep(5)
            generation = documents.pop()
            documents_set.append(documents)
            generated_context.append(generation.page_content)

        print(f"{COLOR['GREEN']}✅: EXECUTION COMPLETED{COLOR['ENDC']}\n")

        context_iter += 1
        return {
            "keys": {
                "documents": documents_set, 
                "generation": generated_context,
                "iterations": iterations,
                "context_iter": context_iter,
                "issue_type": self.issue_type,
                "api_name": self.api_name,
                "documentation": self.documentation,
            }
        }
    

    



    # ========================================================================================================================
    def check_hallucination(self, state: GraphState) -> GraphState:
        print(f"{COLOR['BLUE']}🚀: EXECUTING GRADER: Hallucination and Answer Relevancy Checker.{COLOR['ENDC']}")
        state_dict = state["keys"]
        documents_set = state_dict["documents"]
        generated_context = state_dict["generation"]
        iterations = state_dict["iterations"]
        context_iter = state_dict["context_iter"]

        contexts = []

        for i, docs in enumerate(documents_set):
            documents = "\n".join([d for d in docs])
            documents = Document(page_content=documents)

            hg = graders.hallucination_grader()
            score = hg.invoke({"documents": documents, "generation": generated_context[i]})
            time.sleep(5)
            answer_grounded = score['binary_score']
            if answer_grounded == "no":
                print(f"\t{COLOR['RED']}--- ➡️ DECISION: GENERATED CONTEXT IS NOT GROUNDED ---{COLOR['ENDC']}")
                grade = "no"
            else:
                print(f"\t{COLOR['GREEN']}--- ➡️ DECISION: GENERATED CONTEXT IS GROUNDED ---{COLOR['ENDC']}")
                grade = "yes"
                contexts.append(generated_context[i])

            
        print(f"{COLOR['GREEN']}✅: EXECUTION COMPLETED{COLOR['ENDC']}\n")
        return {
            "keys": {
                "grade": grade,
                "iterations": iterations,
                "context_iter": context_iter,
                "documentation": self.documentation,
            }
        }


    # ========================================================================================================================
    def check_issue_type(self, state: GraphState) -> GraphState:
        state_dict = state["keys"]
        context = state_dict["context"]
        iterations = state_dict["iterations"]
        flag = False

        if self.issue_type == "examples_required":
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
        self.question_id = state_dict["question_id"]
        self.title = state_dict["title"]
        self.question = state_dict["question"]
        self.api_name = state_dict["api_name"]
        self.context = state_dict["context"]
        iterations = state_dict["iterations"]

        try:
            self.documentation = utils.get_documentation(self.api_name)
            # results = tasks.prompt_task(self.issue_type)
            # self.definition = results['definition']
            self.task = "Your task is to generate complete executable code example to address the quesetion body by only using the knowledge provided as 'context'."

            ## Data model
            class code(BaseModel):
                """Code output"""
                prefix: str = Field(description="Description of the problem and approach")
                imports: str = Field(description="Code block import statements")
                code: str = Field(description="Code block not including import statements")

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
            template = template_1.return_template(image)

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
                    input_variables=["context", "title", "question", "documentation", 
                                     "task", "generation", "error"],
                )

                # Chain
                chain = (
                    {
                        "context": lambda _: self.context,
                        "documentation": lambda _: self.documentation,
                        "title": lambda _: self.title,
                        "question": lambda _: self.question,
                        # "api_name": lambda _: self.api_name,
                        # "issue_type": lambda _: self.issue_type,
                        # "definition": lambda _: self.definition,
                        "task": lambda _: self.task,
                        "generation": itemgetter("generation"),
                        "error": itemgetter("error"),
                    }
                    | prompt
                    | llm_with_tool
                    | parser_tool
                )

                code_solution = chain.invoke(
                    {   "context": self.context,
                        "documentation": self.documentation,
                        "title": self.title,
                        "question": self.question,
                        "task": self.task,
                        "generation": str(code_solution[0]),
                        "error": error,
                    }
                )

            else:
                print(f"{COLOR['YELLOW']} {'='*10} 🛠 GENERATE SOLUTION {'='*10} {COLOR['ENDC']}\n")

                # Prompt
                prompt = PromptTemplate(
                    template=template,
                    input_variables=['context', 'title', 'question', 'documentation','task'])

                # Chain
                chain = (
                    {
                        "context": lambda _: self.context,
                        "documentation": lambda _: self.documentation,
                        "title": lambda _: self.title,
                        "question": lambda _: self.question,
                        # "api_name": lambda _: self.api_name,
                        # "issue_type": lambda _: self.issue_type,
                        # "definition": lambda _: self.definition,
                        "task": lambda _: self.task,
                    }
                    | prompt
                    | llm_with_tool
                    | parser_tool
                )

                code_solution = chain.invoke(
                    {
                        "context": self.context,
                        "documentation": self.documentation,
                        "title": self.title,
                        "question": self.question,
                        "task": self.task,
                    }
                )
                
            iterations += 1
            return {
                "keys": {
                    "generation": code_solution,
                    "context": self.context,
                    "iterations": iterations,
                    "question_id": self.question_id,
                    "title": self.title,
                    "question": self.question,
                    # "issue_type": self.issue_type,
                    "api_name": self.api_name,
                    "documentation": self.documentation,
                }
            }
    
        except ValueError as e:
            print(e)


# Function created for HumanEval
    # def generate(self, state: GraphState) -> GraphState:
    #     state_dict = state["keys"]
    #     self.question_id = state_dict["question_id"]
    #     self.title = state_dict["title"]
    #     self.question = state_dict["question"]
    #     self.context = state_dict["context"]
    #     self.documentation = state_dict["documentation"]
    #     # self.problem = state_dict["problem"]
    #     # self.input = state_dict["prompt"]
    #     iterations = state_dict["iterations"]

    #     try:
    #         ## Data model
    #         class code(BaseModel):
    #             """Code output"""
    #             prefix: str = Field(description="Description of the problem and approach")
    #             imports: str = Field(description="Code block import statements")
    #             code: str = Field(description="Code block not including import statements")


    #         llm = ChatOpenAI(temperature=0, model=self.model, streaming=True)
    #         code_tool_oai = convert_to_openai_tool(code)

    #         # LLM with tool and enforce invocation
    #         llm_with_tool = llm.bind(
    #             tools=[code_tool_oai],
    #             tool_choice={"type": "function", "function": {"name": "code"}},
    #         )

    #         # Parser
    #         parser_tool = PydanticToolsParser(tools=[code])

    #         ## Prompt
    #         template = template_1.return_template(image)

    #         ## Generation
    #         if "error" in state_dict:
    #             print(f"{COLOR['RED']} {'='*10} 🔄 RE-GENERATE SOLUTION w/ ERROR FEEDBACK {'='*10} {COLOR['ENDC']}\n")

    #             error = state_dict["error"]
    #             code_solution = state_dict["generation"]

    #             # Update prompt
    #             addendum = """  
    #                     \n --- --- --- \n You previously tried to solve this problem. \n 
    #                     Here is your solution:
    #                     \n --- --- --- \n {generation}  \n --- --- --- \n  
                        
    #                     Here is the resulting error from code execution:  
    #                     \n --- --- --- \n {error}  \n --- --- --- \n 
    #                     Please re-try to answer this.
    #                     Structure your answer with a instructional description of the code solution. \n 
    #                     Then list the imports. And finally list the functioning code block. Structure your answer with a description of the code solution. \n 
    #                     Then list the imports. And finally list the functioning code block.
    #                     \n Here is the user question: \n --- --- --- \n {prompt}
    #                     """
    #             template = template + addendum

    #             # Prompt
    #             prompt = PromptTemplate(
    #                 template=template,
    #                 input_variables=["prompt", "generation", "error"],
    #             )


    #             chain = (
    #                 {
    #                     "prompt": lambda _: self.input,
    #                     "generation": itemgetter("generation"),
    #                     "error": itemgetter("error"),
    #                 }
    #                 | prompt
    #                 | llm_with_tool
    #                 | parser_tool
    #             )

    #             code_solution = chain.invoke(
    #                 {   "context": self.input,
    #                     "generation": str(code_solution[0]),
    #                     "error": error,
    #                 }
    #             )


    #         else:
    #             print(f"{COLOR['YELLOW']} {'='*10} 🛠 GENERATE SOLUTION {'='*10} {COLOR['ENDC']}\n")

    #             # Prompt
    #             prompt = PromptTemplate(
    #                 template=template,
    #                 input_variables=['prompt'])

    #             chain = (
    #                 {
    #                     "prompt": lambda _: self.input,
    #                 }
    #                 | prompt
    #                 | llm_with_tool
    #                 | parser_tool
    #             )


    #             code_solution = chain.invoke(
    #                 {   "prompt": self.input,
    #                 }
    #             )
                
    #         iterations += 1
    #         return {
    #             "keys": {
    #                 # "question": self.question,
    #                 "generation": code_solution,
    #                 "iterations": iterations,
    #                 "problem": self.problem,
    #             }
    #         }
    
    #     except ValueError as e:
    #         print(e)


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
            self.documentation = utils.get_documentation(self.api_name)
            results = tasks.prompt_task(self.issue_type)
            self.definition = results['definition']
            self.task = results['task']

            ## LLM
            llm = ChatOpenAI(temperature=0, model=self.model, streaming=True)


            ## Prompt
            template = template_2.return_template()


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
                | llm
            )

            solution = chain.invoke(
                {
                    "context": context,
                }
            )
            # print(solution.content)
            return {
                "keys": {
                    "generation": solution.content,
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

        error_status = "No imports are required"

        ## State
        print(f"{COLOR['BLUE']}⏳ CHECKING CODE IMPORTS{COLOR['ENDC']}")
        state_dict = state["keys"]
        code_solution = state_dict["generation"]
        imports = code_solution[0].imports
        iterations = state_dict["iterations"]
        # problem = state_dict["problem"]

        # Attempt to execute the imports
        sb = sandbox.run(imports)
        output, error = sb.stdout.read(), sb.stderr.read()

        if error:
            # if error_status in str(error):
            #     print(f"\t{COLOR['RED']}--- ❌ {error_status} ---{COLOR['ENDC']}\n")
            #     error = error_status
            # else:
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
                "iterations": iterations,
                "context": self.context,
                "iterations": iterations,
                "question_id": self.question_id,
                "title": self.title,
                "question": self.question,
                "api_name": self.api_name,
                "documentation": self.documentation,
                "error": error,
                # "problem": problem,
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
        results = defaultdict(list)

        ## State
        print(f"{COLOR['BLUE']}⏳ CHECKING CODE EXECUTION {COLOR['ENDC']}")
        state_dict = state["keys"]
        # context = state_dict["context"]
        code_solution = state_dict["generation"]
        prefix = code_solution[0].prefix
        imports = code_solution[0].imports
        code = code_solution[0].code
        iterations = state_dict["iterations"]
        # prev_error = state_dict["error"]
        # problem = state_dict["problem"]

        # error_status = "No imports are required"
        # if prev_error == error_status:
        #     code_block = code
        # else:
            
        code_block = imports + "\n" + code

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

            # pass_at_1_mean = 0
            # bleu_score_1 = 0
            # bleu_score_2 = 0
            # bleu_score_3 = 0
            # bleu_score_4 = 0
        else:
            print(f"\t{COLOR['GREEN']}--- ✅ CODE BLOCK CHECK: SUCCESS ---{COLOR['ENDC']}\n")
            error = "None"

            # result = check_correctness(problem, code_block, timeout=10)
            # results[result["task_id"]].append((result["completion_id"], result))

            # total, correct = [], []
            # for result in results.values():
            #     passed = [r[1]["passed"] for r in result]
            #     total.append(len(passed))
            #     correct.append(sum(passed))

            # total = np.array(total)
            # correct = np.array(correct)

            # # Compute pass@1
            # pass_at_1 = correct / total
            # pass_at_1_mean = pass_at_1.mean()
            # # print(f"pass@1: {pass_at_1.mean()}")

            # reference_tokens = nltk.word_tokenize(problem['canonical_solution'])
            # generated_tokens = nltk.word_tokenize(code_block)
            # # smoothie = SmoothingFunction().method4
            # bleu_score_1 = sentence_bleu([reference_tokens], generated_tokens, weights=(1, 0, 0, 0))
            # bleu_score_2 = sentence_bleu([reference_tokens], generated_tokens, weights=(0, 1, 0, 0))
            # bleu_score_3 = sentence_bleu([reference_tokens], generated_tokens, weights=(0, 0, 1, 0))
            # bleu_score_4 = sentence_bleu([reference_tokens], generated_tokens, weights=(0, 0, 0, 1))

            # print(f"pass@1: {pass_at_1_mean}")
            # print(f"bleu_score_1: {bleu_score_1}")
            # print(f"bleu_score_2: {bleu_score_2}")
            # print(f"bleu_score_3: {bleu_score_3}")
            # print(f"bleu_score_4: {bleu_score_4}")

        return {
            "keys": {
                "question": self.question,
                "generation": code_solution,
                "error": error,
                "prefix": prefix,
                "imports": imports,
                "code": code,
                "iterations": iterations,
                "question_id": self.question_id,
                "title": self.title,
                "question": self.question,
                "api_name": self.api_name,
                "documentation": self.documentation,
                # "task_id": problem["task_id"],
                # "problem": problem,
                # "pass_at_1": pass_at_1_mean,
            }
        }


    # ========================================================================================================================
    def ragas_eval(self, state: GraphState) -> GraphState:
        state_dict = state["keys"]
        contexts = state_dict["context"]
        iterations = state_dict["iterations"]

        if self.issue_type == "examples_required":
            solution = state_dict["generation"]
            code_solution = state_dict["generation"][0]
            error = state_dict["error"]
            prefix = code_solution.prefix
            imports = code_solution.imports
            code = code_solution.code
            answer = "\n".join([prefix, imports, code])

        else:
            answer = state_dict["generation"]

        metrics = [
            answer_relevancy,
            faithfulness,
            context_precision,
            context_relevancy,
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
        print(f"\t{COLOR['BLUE']}➡️ Context Precision      : {results_df['context_precision'][0]} {COLOR['ENDC']}")
        print(f"\t{COLOR['BLUE']}➡️ Context Relevancy   : {results_df['context_relevancy'][0]} {COLOR['ENDC']}")
        # print(f"\t{COLOR['BLUE']}➡️ Answer Correctness  : {results_df['answer_correctness'][0]} {COLOR['ENDC']}")
        print(f"\t{COLOR['BLUE']}➡️ Answer Similarity   : {results_df['answer_similarity'][0]} {COLOR['ENDC']}")
        print(f"\t{COLOR['BLUE']}➡️ Answer Relevancy    : {results_df['answer_relevancy'][0]} {COLOR['ENDC']}")

        if self.issue_type == "examples_required":
            return {
                "keys": {
                    "generation": solution,
                    "error": error,
                    "prefix": prefix,
                    "imports": imports,
                    "context": contexts,
                    "iterations": iterations,
                    "code": code,
                    "question_id": self.question_id,
                    "question": self.question,
                    "ground_truth": self.ground_truth,
                    "issue_type": self.issue_type,
                    "api_name": self.api_name,
                    "documentation": self.documentation,
                    "intent": self.intention,
                    "urls": self.so_answers,
                    "course_urls": self.course_urls,
                    "faithfulness": results_df["faithfulness"][0],
                    "context_precision": results_df["context_precision"][0],
                    "context_relevancy": results_df["context_relevancy"][0],
                    # "answer_correctness": results_df["answer_correctness"][0],
                    "answer_similarity": results_df["answer_similarity"][0],
                    "answer_relevancy": results_df["answer_relevancy"][0],
                }
            }
        else:
                return {
                "keys": {
                    "generation": answer,
                    "context": contexts,
                    "iterations": iterations,
                    "question_id": self.question_id,
                    "question": self.question,
                    "ground_truth": self.ground_truth,
                    "issue_type": self.issue_type,
                    "api_name": self.api_name,
                    "documentation": self.documentation,
                    "intent": self.intention,
                    "urls": self.so_answers,
                    "course_urls": self.course_urls,
                    "faithfulness": results_df["faithfulness"][0],
                    "context_precision": results_df["context_precision"][0],
                    "context_relevancy": results_df["context_relevancy"][0],
                    # "answer_correctness": results_df["answer_correctness"][0],
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
        print(f"\n{COLOR['YELLOW']}🏁 FINISHING {COLOR['ENDC']}\n")
        state_dict = state["keys"]
        # issue_type = state_dict["issue_type"]

        if self.debug:
            response = extract_eval_response(state)
        else:
            response = extract_response(state)

        return {"keys": {"response": response}}


# ========================================================================================================================
# def extract_eval_response(state: GraphState) -> str:
#     # examples_required = [
#     #         "Documentation Replication on Other Examples", 
#     #         "Documentation Replicability", 
#     #         "Inadequate Examples"]
    
#     state_dict = state["keys"]
#     issue_type = state_dict["issue_type"]
#     documentation = state_dict["documentation"]
#     urls = state_dict["urls"]
#     course_urls = state_dict["course_urls"]
#     api_name = state_dict["api_name"]

#     if issue_type == "examples_required":
#         code_solution = state_dict["generation"][0]
#         prefix = code_solution.prefix
#         imports = code_solution.imports
#         code = code_solution.code
#         answer = "\n".join([prefix, imports, code])
#         output = output_template.return_template1(api_name, prefix, imports, code, urls)
#         final = documentation + output
#         if state_dict["error"] == "None":
#             error = "False"
#         else:
#             error = "True"
#     elif issue_type == "description_only":
#         answer = state_dict["generation"]
#         output = output_template.return_template2(api_name, answer, urls)
#         final = documentation + output
#         error = "False"
#     else:
#         output = output_template.return_template3(api_name, urls, course_urls)
#         final = documentation + output
#         error = "False"

#     ragas = {
#         "question_id": str(state_dict["question_id"]),
#         "question": str(state_dict["question"]),
#         "intent": str(state_dict["intent"]),
#         "ground_truth": str(state_dict["ground_truth"]),
#         "context": str(state_dict["context"]) if issue_type != "additional_resources" else "N/A",
#         "final" : str(final),
#         # "so_urls": str(urls),
#         "faithfulness": str(state_dict["faithfulness"]) if issue_type != "additional_resources" else "N/A",
#         "context_precision": str(state_dict["context_precision"]) if issue_type != "additional_resources" else "N/A",
#         "context_relevancy": str(state_dict["context_relevancy"]) if issue_type != "additional_resources" else "N/A",
#         # "answer_correctness": str(state_dict["answer_correctness"]),
#         "answer_similarity": str(state_dict["answer_similarity"]) if issue_type != "additional_resources" else "N/A",
#         "answer_relevancy": str(state_dict["answer_relevancy"]) if issue_type != "additional_resources" else "N/A",
#         "execution_error": error,
#     }
#     # print(ragas)
#     return json.dumps(ragas)


def extract_eval_response(state: GraphState) -> str:    
    state_dict = state["keys"]

    code_solution = state_dict["generation"][0]
    prefix = code_solution.prefix
    imports = code_solution.imports
    code = code_solution.code
    answer = "\n".join([prefix, imports, code])

    if state_dict["error"] == "None":
        execution = "Success"
    else:
        execution = "Failed"

    output = {
        # "task_id": state_dict["task_id"],
        # "problem": state_dict["problem"],
        # "question": state_dict["question"],
        "question_id": state_dict["question_id"],
        "title": state_dict["title"],
        "question": state_dict["question"],
        "api_name": state_dict["api_name"],
        "documentation": state_dict["documentation"],
        "generated_answer": answer,
        "execution_status": execution,
        "iterations": state_dict["iterations"],
        # "pass_at_1": state_dict["pass_at_1"],
        # "bleu_score_1": state_dict["bleu_score_1"],
        # "bleu_score_2": state_dict["bleu_score_2"],
        # "bleu_score_3": state_dict["bleu_score_3"],
        # "bleu_score_4": state_dict["bleu_score_4"],
    }

    return json.dumps(output)



def extract_response(state: GraphState) -> str:


    state_dict = state["keys"]
    issue_type = state_dict["issue_type"]
    documentation = state_dict["documentation"]

    if issue_type == "examples_required":
        api_name = state_dict["api_name"]
        # urls = state_dict["urls"]
        code_solution = state_dict["generation"][0]
        prefix = code_solution.prefix
        imports = code_solution.imports
        code = code_solution.code

        output = output_template.return_template1(api_name, prefix, imports, code)
        final = documentation + output

    elif issue_type == "description_only":
        documentation = state_dict["documentation"]
        solution = state_dict["generation"][0]

        output = output_template.return_template2(api_name, solution)
        final = documentation + output
    
    else:
        pass
    return final