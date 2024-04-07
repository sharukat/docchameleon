# ==========================================================================================================================
# IMPORT DEPENDENCIES
# ==========================================================================================================================
import pprint
import retrieval_nodes
import retrieval_edges
from lib.common import RagGraphState
from langgraph.graph import StateGraph



# ==========================================================================================================================
# MAIN CODE
# ==========================================================================================================================

def rag_construct_graph():
    graph = StateGraph(RagGraphState)

    graph_nodes = retrieval_nodes.Nodes()
    for key, value in graph_nodes.node_map.items():
        graph.add_node(key, value)

    graph = retrieval_edges.enrich(graph)
    graph.set_entry_point(key="generate")
    graph.set_finish_point(key="finish")
    return graph


def relevant_context_retriever(question: str):
    graph = rag_construct_graph().compile()
    inputs = {"question": question, "iterations": 0}
    context = graph.invoke(inputs)
    # for output in graph.stream(inputs):
    #     for key, value in output.items():
    #         pprint.pprint(f"Node '{key}':")
    # context = value["generation"]
    return context['generation']

# question = "The user is seeking guidance on how to correctly migrate a TensorFlow 1 code snippet to TensorFlow 2, specifically aiming to replicate the exact shape of a tensor created with tf.compat.v1.placeholder in TensorFlow 1 using tf.keras.Input in TensorFlow 2."
# context = relevant_context_retriever(question)
# print(context)
# print(type(context))