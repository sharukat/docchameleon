# ==========================================================================================================================
# IMPORT DEPENDENCIES
# ==========================================================================================================================
# from langchain_cohere import ChatCohere, CohereRagRetriever
# # from langchain_community.retrievers import CohereRagRetriever

# import os
# from dotenv import load_dotenv
# load_dotenv(dotenv_path="/Users/sharukat/Documents/ResearchYU/Code/doc-customizer-llm/doc_customizer_llm/.env")
# os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY") 


# # ==========================================================================================================================
# # MAIN CODE
# # ==========================================================================================================================

# def retrieve(query):
#     rag = CohereRagRetriever(llm=ChatCohere(), connectors=[{"id": "web-search"}])
#     documents = rag.get_relevant_documents(query)
#     generation = documents.pop()
#     generation = generation.page_content
#     return generation

# question = "The user is seeking guidance on how to correctly migrate a TensorFlow 1 code snippet to TensorFlow 2, specifically aiming to replicate the exact shape of a tensor created with tf.compat.v1.placeholder in TensorFlow 1 using tf.keras.Input in TensorFlow 2."
# context = retrieve(question)
# print(context)
# print(type(context))