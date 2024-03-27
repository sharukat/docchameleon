"""This module defines our agent and attaches it to the Modal Stub.

Our agent is defined as a graph: a collection of nodes and edges,
where nodes represent actions and edges represent transitions between actions.

The meat of the logic is therefore in the edges and nodes modules.

We have a very simple "context-stuffing" retrieval approach in the retrieval module.
Replace this with something that retrieves your documentation and adjust the prompts accordingly.

You can test the agent from the command line with `modal run agent.py --question` followed by your query"""

import edges
import nodes
from common import stub


@stub.local_entrypoint()
def main(question: str, debug: bool = False):
    """Sends a question to the TensorFlow code generation agent.

    Switch to debug mode for shorter context and smaller model."""
    if debug:
        if question == "How do I build a RAG pipeline?":
            question = "gm king, how are you?"
    if question != None:
        print(go.remote(question, debug=debug)["keys"]["response"])


@stub.function(gpu="any")
def go(question: str, debug: bool = False):
    """Compiles the LCEL code generation agent graph and runs it, returning the result."""
    graph = construct_graph(debug=debug)
    runnable = graph.compile()
    result = runnable.invoke(
        {"keys": {"question": question, "iterations": 0}},
        config={"recursion_limit": 3},
    )

    return result


def construct_graph(debug=False):
    from common import GraphState
    from langgraph.graph import StateGraph

    graph = StateGraph(GraphState)

    # attach our nodes to the graph
    graph_nodes = nodes.Nodes(debug=debug)
    for key, value in graph_nodes.node_map.items():
        graph.add_node(key, value)

    # construct the graph by adding edges
    graph = edges.enrich(graph)

    # set the starting and ending nodes of the graph
    graph.set_entry_point(key="generate")
    graph.set_finish_point(key="finish")

    return graph