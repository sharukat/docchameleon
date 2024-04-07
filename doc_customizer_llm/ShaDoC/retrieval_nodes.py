import graders
from lib.common import RagGraphState
from lib.config import COLOR
from langchain.schema import Document
from langchain_community.chat_models import ChatCohere
from langchain_community.retrievers import CohereRagRetriever

import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="/Users/sharukat/Documents/ResearchYU/Code/doc-customizer-llm/doc_customizer_llm/.env")
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")


class Nodes:
    def __init__(self):
        self.node_map = {
            "generate": self.generate,
            "check_context_relevancy": self.check_context_relevancy,
            "check_hallucination": self.check_hallucination,
            "check_answer_relevancy": self.check_answer_relevancy,
            "finish": self.finish,
        }


    def generate(self, state: RagGraphState) -> RagGraphState:
        """
        Retrieve relevant context from web using Cohere RAG Retriever
        Args:
            state (dict): The current graph state
        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print(f"{COLOR['BLUE']}🚀: EXECUTING RETRIEVER: Cohere RAG Retriever{COLOR['ENDC']}")
        question = state["question"]
        iter = state["iterations"]

        # RAG generation
        print("Step 1")
        rag = CohereRagRetriever(llm=ChatCohere(model="command-r"), connectors=[{"id": "web-search"}])
        print("Step 2")
        documents = rag.get_relevant_documents(question)
        print("Step 3")
        generation = documents.pop()
        print("Step 4")
        generation = generation.page_content
        print(f"{COLOR['GREEN']}✅: EXECUTION COMPLETED{COLOR['ENDC']}\n")

        iter = iter + 1
        return {
            "documents": documents, 
            "question": question, 
            "generation": generation,
            "iterations": iter,}

    def check_context_relevancy(self, state: RagGraphState) -> RagGraphState:
        """
        Determines whether the retrieved documents are relevant to the question.
        Args:
            state (dict): The current graph state
        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """
        print(f"{COLOR['BLUE']}🚀: EXECUTING GRADER: Context vs Question Checker.{COLOR['ENDC']}")
        question = state["question"]
        documents = state["documents"]
        iter = state["iterations"]

        # Score each doc
        filtered_docs = []
        for d in documents:
            rg = graders.retrieval_grader()
            score = rg.invoke({"question": question, "document": d.page_content})
            grade = score.binary_score
            if grade == "yes":
                print(f"\t{COLOR['GREEN']}--- ➡️ GRADE: DOCUMENT RELEVANT ---{COLOR['ENDC']}")
                filtered_docs.append(d)
            else:
                print(f"\t{COLOR['RED']}--- ➡️ GRADE: DOCUMENT NOT RELEVANT ---{COLOR['ENDC']}")
                continue
        print(f"{COLOR['GREEN']}✅: EXECUTION COMPLETED{COLOR['ENDC']}\n")
        
        return {
            "documents": filtered_docs, 
            "question": question,
            "iterations": iter,
            }



    def check_hallucination(self, state: RagGraphState) -> RagGraphState:
        print(f"{COLOR['BLUE']}🚀: EXECUTING GRADER: Hallucination Checker.{COLOR['ENDC']}")
        documents = state["documents"]
        generation = state["generation"]
        iter = state["iterations"]

        documents = "\n".join([d.page_content for d in documents])
        documents = Document(page_content=documents)

        hg = graders.hallucination_grader()
        score = hg.invoke({"documents": documents, "generation": generation})
        grade = score.binary_score

        print(f"{COLOR['GREEN']}✅: EXECUTION COMPLETED{COLOR['ENDC']}\n")
        
        return {
            "hallucinations": grade,
            "iterations": iter,
            }
    


    def check_answer_relevancy(self, state: RagGraphState) -> RagGraphState:
        print(f"{COLOR['BLUE']}🚀: EXECUTING GRADER: Generation vs Question Checker.{COLOR['ENDC']}")
        question = state["question"]
        generation = state["generation"]
        iter = state["iterations"]

        ag = graders.answer_grader()
        score = ag.invoke({"question": question,"generation": generation})
        grade = score.binary_score
        print(f"{COLOR['GREEN']}✅: EXECUTION COMPLETED{COLOR['ENDC']}\n")

        return {
            "answer_relevancy": grade, 
            "iterations": iter,
            }

        
    def finish(self, state: RagGraphState) -> dict:
        """
        Finish the graph

        Returns:
            dict: Final result
        """
        generation = state["generation"]
        print(f"\n{COLOR['YELLOW']}🏁 FINISHING RETRIEVAL {COLOR['ENDC']}")
        return {"generation": generation}