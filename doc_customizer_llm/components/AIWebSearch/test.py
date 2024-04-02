from cohere_rag_retriever import relevant_context_retriever
from search import course_urls_retriever

# ==========================================================================================================================
# TEST EXECUTIONS
# ==========================================================================================================================

query = """
    The user is seeking guidance on how to correctly migrate TensorFlow 1 code to TensorFlow 2, specifically on how to replicate 
    the functionality and output of the `tf.compat.v1.placeholder` with `tf.keras.Input` in TensorFlow 2. They are encountering 
    an issue where the shapes of the tensors produced by the two methods are not matching, and they are looking for assistance in 
    resolving this discrepancy to achieve the same tensor shape in TensorFlow 2 as they had in TensorFlow 1.
    """

model_response, context, urls = relevant_context_retriever(query)
