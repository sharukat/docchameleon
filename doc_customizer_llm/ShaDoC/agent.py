import edges
import nodes
from lib.common import stub, GraphState
import lib.utils as utils
from langgraph.graph import StateGraph


def construct_graph(debug=False):
    graph = StateGraph(GraphState)

    # attach our nodes to the graph
    graph_nodes = nodes.Nodes(debug=debug)
    for key, value in graph_nodes.node_map.items():
        graph.add_node(key, value)

    # construct the graph by adding edges
    graph = edges.enrich(graph)

    # set the starting and ending nodes of the graph
    graph.set_entry_point(key="check_issue_type")
    graph.set_finish_point(key="finish")

    return graph