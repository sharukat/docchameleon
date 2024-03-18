from cohere_rag_retriever import relevant_context_retriever

# ==========================================================================================================================
# TEST EXECUTIONS
# ==========================================================================================================================

query = """
    The user is trying to migrate code from TensorFlow 1 to TensorFlow 2. Specifically, they want to find the equivalent of 
    tf.compat.v1.placeholder() in TensorFlow 2 that produces a tensor with the same shape, as the direct replacement 
    tf.keras.Input() seems to result in a different shape.
    """

output = relevant_context_retriever(query)
