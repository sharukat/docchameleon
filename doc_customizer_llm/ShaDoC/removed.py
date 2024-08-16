# ========================================================================================================================
# def check_hallucination_and_answer_relevancy(self, state: GraphState) -> GraphState:
#     print(f"{COLOR['BLUE']}🚀: EXECUTING GRADER: Hallucination and Answer Relevancy Checker.{COLOR['ENDC']}")
#     state_dict = state["keys"]
#     documents_set = state_dict["documents"]
#     generated_context = state_dict["generation"]
#     iterations = state_dict["iterations"]
#     context_iter = state_dict["context_iter"]

#     contexts = []

#     for i, docs in enumerate(documents_set):
#         documents = "\n".join([d for d in docs])
#         documents = Document(page_content=documents)

#         hg = graders.hallucination_grader()
#         score = hg.invoke({"documents": documents, "generation": generated_context[i]})
#         time.sleep(10)
#         answer_grounded = score['binary_score']
#         if answer_grounded == "no":
#             print(f"\t{COLOR['RED']}--- ➡️ DECISION: GENERATED CONTEXT IS NOT GROUNDED ---{COLOR['ENDC']}")
#             grade = "no"
#         else:
#             print(f"\t{COLOR['GREEN']}--- ➡️ DECISION: GENERATED CONTEXT IS GROUNDED ---{COLOR['ENDC']}")
#             grade = "yes"
#             contexts.append(generated_context[i])

#         # ag = graders.answer_grader()
#         # score = ag.invoke({"question": self.question,"generation": generation})
#         # answer_relevancy = score['binary_score']
#         # if answer_relevancy == "yes":
#         #     print(f"\t{COLOR['GREEN']}--- ➡️ DECISION: LLM GENERATION RESOLVES THE QUESTION ---{COLOR['ENDC']}")
#         #     grade = "yes"
#         # else:
#         #     grade = "no"
#         #     print(f"\t{COLOR['RED']}--- ➡️ DECISON: LLM GENERATION DOES NOT RESOLVES THE QUESTION. Re-TRY ---{COLOR['ENDC']}")
#     print(f"{COLOR['GREEN']}✅: EXECUTION COMPLETED{COLOR['ENDC']}\n")
#     context = "\n\n".join([con for con in contexts])
#     return {
#         "keys": {
#             "documents": documents, 
#             "context": context,
#             "grade": grade,
#             "iterations": iterations,
#             "context_iter": context_iter,
#             "issue_type": self.issue_type,
#             "api_name": self.api_name,
#             "documentation": self.documentation,
#         }
#     }




# ========================================================================================================================
# def check_context_relevancy(self, state: GraphState) -> GraphState:
#     """
#     Determines whether the retrieved documents are relevant to the question.
#     Args:
#         state (dict): The current graph state
#     Returns:
#         state (dict): Updates documents key with only filtered relevant documents
#     """
#     print(f"{COLOR['BLUE']}🚀: EXECUTING GRADER: Context vs Question Checker.{COLOR['ENDC']}")

#     state_dict = state["keys"]
#     documents_sets = state_dict["documents"]
#     generationed_context = state_dict["generation"]
#     iterations = state_dict["iterations"]
#     context_iter = state_dict["context_iter"]

#     # Score each doc
#     filtered_docs = []
#     for documents in documents_sets:
#         docs = []
#         for d in documents:
#             rg = graders.retrieval_grader()
#             score = rg.invoke({"question": self.question, "document": d.page_content})
#             time.sleep(10)
#             grade = score['binary_score']
#             if grade == "yes":
#                 print(f"\t{COLOR['GREEN']}--- ➡️ GRADE: DOCUMENT RELEVANT ---{COLOR['ENDC']}")
#                 docs.append(d.page_content)
#             else:
#                 print(f"\t{COLOR['RED']}--- ➡️ GRADE: DOCUMENT NOT RELEVANT ---{COLOR['ENDC']}")
#                 continue
#         filtered_docs.append(docs)

#     print(f"{COLOR['GREEN']}✅: EXECUTION COMPLETED{COLOR['ENDC']}\n")
    
#     return {
#         "keys": {
#             "documents": filtered_docs, 
#             "generation": generationed_context,
#             "iterations": iterations,
#             "context_iter": context_iter,
#             "issue_type": self.issue_type,
#             "api_name": self.api_name,
#             "documentation": self.documentation,
#         }
#     }




# def check_additional_resources(self, state: GraphState) -> GraphState:
#     state_dict = state["keys"]
#     iterations = state_dict["iterations"]
#     context_iter = state_dict["context_iter"]
#     intent = state_dict["intent"]
#     flag = False

#     if self.issue_type == "additional_resources":
#         flag = True

#     return {
#         "keys": {
#             "iterations": iterations,
#             "context_iter": context_iter,
#             "example_required": flag,
#             "issue_type": self.issue_type,
#             "api_name": self.api_name,
#             "documentation": self.documentation,
#             "intent":intent,
#         }
#     }




        # self.examples_required = [
        #     "Documentation Replication on Other Examples", 
        #     "Documentation Replicability", 
        #     "Inadequate Examples"]
        
        # self.description_only = [
        #     "Documentation Ambiguity",
        #     "Documentation Completeness"]




# keywords = results['keywords']
# so_relevant_answers = stackoverflow.retrieval(self.title, self.question, keywords)
# if so_relevant_answers is not None:
#     self.so_answers = so_relevant_answers['urls']

# if self.issue_type == 'additional_resources':
#     search_results = search.course_urls_retriever(self.intention)
#     course_urls = search_results['urls']
#     if not course_urls:
#         self.course_urls = utils.remove_broken_urls(course_urls)