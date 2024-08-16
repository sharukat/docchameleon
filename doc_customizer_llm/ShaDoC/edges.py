"""Defines functions that transition our agent from one state to another."""

from typing import Callable
from lib.common import GraphState
from lib.config import COLOR

EXPECTED_NODES = [
    # "intent_soanswers_courses",
    # "check_additional_resources",
    # "context_retrieval",
    # "check_context_relevancy",
    # "check_hallucination_and_answer_relevancy",
    # "check_issue_type",
    "generate",
    # "generate_without_examples",
    "check_code_imports",
    "check_code_execution",
    # "ragas_eval",
    "finish",
]


def enrich(graph):
    """Adds transition edges to the graph."""

    for node_name in set(EXPECTED_NODES):
        assert node_name in graph.nodes, f"Node {node_name} not found in graph"

    # graph.add_edge("intent_soanswers_courses", "check_additional_resources")
    # graph.add_conditional_edges(
    #     "check_additional_resources",
    #     EDGE_MAP["decide_context_or_finish"],
    #     {
    #         "finish":"finish",
    #         "context_retrieval":"context_retrieval",
    #     }
    # )

    # graph.add_edge("context_retrieval", "check_context_relevancy")
    # graph.add_conditional_edges(
    #     "check_context_relevancy",
    #     EDGE_MAP["decide_context_relevancy"],
    #     {
    #         "check_hallucination_and_answer_relevancy":"check_hallucination_and_answer_relevancy",
    #         "context_retrieval":"context_retrieval",
    #     }
    # )
    # graph.add_conditional_edges(
    #     "check_hallucination_and_answer_relevancy",
    #     EDGE_MAP["decide_context_retrieval"],
    #     {
    #         "context_retrieval":"context_retrieval",
    #         "check_issue_type":"check_issue_type",
    #     }
    # )
    # End of first iterative graph
    # graph.add_conditional_edges(
    #     "check_issue_type",
    #     EDGE_MAP["decide_example_requirement"],
    #     { 
    #         "generate":"generate",
    #         "generate_without_examples":"generate_without_examples",
    #     },
    # )

    graph.add_edge("generate", "check_code_imports")
    # graph.add_edge("generate_without_examples", "ragas_eval")
    graph.add_conditional_edges(
        "check_code_imports",
        EDGE_MAP["decide_to_check_code_exec"],
        {
            "check_code_execution": "check_code_execution",
            "generate": "generate",
        },
    )
    # graph.add_conditional_edges(
    #     "check_code_execution",
    #     EDGE_MAP["decide_to_finish"],
    #     {
    #         "ragas_eval": "ragas_eval",
    #         "generate": "generate",
    #     },
    # )

    graph.add_conditional_edges(
        "check_code_execution",
        EDGE_MAP["decide_to_finish"],
        {
            "finish": "finish",
            "generate": "generate",
        },
    )
    # graph.add_edge("ragas_eval", "finish")

    
    return graph



def decide_context_relevancy(state: GraphState) -> str:
    state_dict = state["keys"]
    filtered_documents = state_dict["documents"]
    if not filtered_documents:
        print(f"\t{COLOR['RED']}--- ➡️ DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, EXECUTE RAG AGAIN ---{COLOR['ENDC']}\n")
        return "context_retrieval"
    else:
        return "check_hallucination_and_answer_relevancy"
    

def decide_context_retrieval(state: GraphState) -> str:
    print(f"{COLOR['BLUE']}❓ DECIDE TO PERFORM CONTEXT RETRIEVAL AGAIN {COLOR['ENDC']}")
    print(f"{COLOR['BLUE']}{'-'*60}{COLOR['ENDC']}")
    state_dict = state["keys"]
    grade = state_dict["grade"]
    context_iter = state_dict["context_iter"]

    if grade == "yes" or context_iter >=3:
        print(f"\t{COLOR['GREEN']}--- ➡️ DECISION: CONTEXT RETRIEVAL SATISFIES ALL CONDITIONS ---{COLOR['ENDC']}\n")
        return "check_issue_type"
    else:
        print(f"\t{COLOR['RED']}--- ➡️ DECISION: CONTEXT RETRIEVAL DOES NOT STATISFY ALL CONDITIONS. RE-TRY! ---{COLOR['ENDC']}\n")
        return "context_retrieval"
    
def decide_context_or_finish(state: GraphState) -> str:
    state_dict = state["keys"]
    flag = state_dict["example_required"]
    if flag == "3":
        print(f"\t{COLOR['GREEN']}--- ➡️ DECISION: ADDITIONAL RESOURCES ONLY ---{COLOR['ENDC']}\n")
        return "finish"
    else:
        return "context_retrieval"

def decide_example_requirement(state: GraphState) -> str:
    print(f"{COLOR['BLUE']}❓ DECIDE TO PROVIDE EXAMPLES WITH DESCRIPTION OR JUST DESCRIPTION {COLOR['ENDC']}")
    print(f"{COLOR['BLUE']}{'-'*20}{COLOR['ENDC']}")
    state_dict = state["keys"]
    flag = state_dict["example_required"]
    if flag == True:
        print(f"\t{COLOR['GREEN']}--- ➡️ DECISION: PROVIDE CODE WITH DESCRIPTION ---{COLOR['ENDC']}\n")
        return "generate"
    else:
        print(f"\t{COLOR['GREEN']}--- ➡️ DECISION: PROVIDE ONLY DESCRIPTION ---{COLOR['ENDC']}\n")
        return "generate_without_examples"


def decide_to_check_code_exec(state: GraphState) -> str:
    """
    Determines whether to test code execution, or re-try answer generation.

    Args:
    state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print(f"{COLOR['BLUE']}❓ DECIDE TO TEST CODE EXECUTION {COLOR['ENDC']}")
    print(f"{COLOR['BLUE']}{'-'*20}{COLOR['ENDC']}")
    state_dict = state["keys"]
    error = state_dict["error"]

    if error == "None":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(f"\t{COLOR['GREEN']}--- ➡️ DECISION: TEST CODE EXECUTION ---{COLOR['ENDC']}\n")
        return "check_code_execution"
    # elif error == "No imports are required":
    #     print(f"\t{COLOR['GREEN']}--- ➡️ DECISION: TEST CODE EXECUTION ---{COLOR['ENDC']}\n")
    #     return "check_code_execution"
    else:
        # We have relevant documents, so generate answer
        print(f"\t{COLOR['RED']}--- 🔄 DECISION: RE-TRY SOLUTION---{COLOR['ENDC']}\n")
        return "generate"


def decide_to_finish(state: GraphState) -> str:
    """
    Determines whether to finish (re-try code 3 times).

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print(f"{COLOR['BLUE']}❓ DECIDE TO TEST CODE EXECUTION {COLOR['ENDC']}")
    print(f"{COLOR['BLUE']}{'-'*20}{COLOR['ENDC']}")
    state_dict = state["keys"]
    error = state_dict["error"]
    iter = state_dict["iterations"]

    if error == "None" or iter >= 1:
        print(f"\t{COLOR['GREEN']}--- ✅ DECISION: FINISH ---{COLOR['ENDC']}\n")
        return "finish"
    else:
        print(f"\t{COLOR['RED']}--- 🔄 DECISION: RE-TRY SOLUTION ---{COLOR['ENDC']}\n")
        return "generate"


EDGE_MAP: dict[str, Callable] = {
    "decide_context_relevancy":decide_context_relevancy,
    "decide_context_or_finish":decide_context_or_finish,
    "decide_context_retrieval":decide_context_retrieval,
    "decide_example_requirement":decide_example_requirement,
    "decide_to_check_code_exec": decide_to_check_code_exec,
    "decide_to_finish": decide_to_finish,
}