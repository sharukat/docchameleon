from lib.common import RagGraphState
from typing import Callable
from lib.config import COLOR
from langgraph.graph import END

EXPECTED_NODES = [
    "generate",
    "check_context_relevancy",
    "check_hallucination",
    "check_answer_relevancy",
    "finish",
]

def enrich(graph):
    """Adds transition edges to the graph."""

    for node_name in set(EXPECTED_NODES):
        assert node_name in graph.nodes, f"Node {node_name} not found in graph"

    graph.add_edge("generate", "check_context_relevancy")
    graph.add_conditional_edges(
        "check_context_relevancy",
        EDGE_MAP["decide_context_relevancy"],
        {
            "check_hallucination":"check_hallucination",
            "generate":"generate",
        }
    )
    graph.add_conditional_edges(
        "check_hallucination",
        EDGE_MAP["decide_hallucination"],
        {
            "check_answer_relevancy":"check_answer_relevancy",
            "generate":"generate",
        }
    )
    graph.add_conditional_edges(
        "check_answer_relevancy",
        EDGE_MAP["decide_answer_relevancy"],
        {
            "generate":"generate",
            "finish":"finish",
        }
    )
    
    return graph




def decide_context_relevancy(state: RagGraphState) -> str:
    filtered_documents = state["documents"]
    if not filtered_documents:
        print(f"\t{COLOR['RED']}--- ➡️ DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, EXECUTE RAG AGAIN ---{COLOR['ENDC']}\n")
        return "generate"
    else:
        return "check_hallucination"
    
def decide_hallucination(state: RagGraphState) -> str:
    print(f"{COLOR['BLUE']}❓ DECIDE TO GRADE THE ANSWER FOR HALLUCINATION {COLOR['ENDC']}")
    print(f"{COLOR['BLUE']}{'-'*60}{COLOR['ENDC']}")
    hallucination = state["hallucinations"]
    if hallucination == "no":
        print(f"\t{COLOR['RED']}--- ➡️ DECISION: LLM GENERATION IS NOT GROUNDED ---{COLOR['ENDC']}\n")
        return "generate"
    else:
        print(f"\t{COLOR['GREEN']}--- ➡️ DECISION: LLM GENERATION IS GROUNDED ---{COLOR['ENDC']}\n")
        return "check_answer_relevancy"
    
def decide_answer_relevancy(state: RagGraphState) -> str:
    print(f"{COLOR['BLUE']}❓ DECIDE TO PROVIDE THE ANSWER RELEVANCY TO THE QUESTION {COLOR['ENDC']}")
    print(f"{COLOR['BLUE']}{'-'*60}{COLOR['ENDC']}")
    answer_relevancy = state["answer_relevancy"]
    iter = state["iterations"]

    if answer_relevancy == "yes" or iter >=4:
        print(f"\t{COLOR['GREEN']}--- ➡️ DECISION: LLM GENERATION RESOLVES THE QUESTION ---{COLOR['ENDC']}\n")
        return "finish"
        
    else:
        print(f"\t{COLOR['RED']}--- ➡️ DECISION: LLM GENERATION DOES NOT RESOLVES THE QUESTION. Re-TRY ---{COLOR['ENDC']}\n")
        return "generate"
    
    
EDGE_MAP: dict[str, Callable] = {
    "decide_context_relevancy":decide_context_relevancy,
    "decide_hallucination":decide_hallucination,
    "decide_answer_relevancy":decide_answer_relevancy,
}