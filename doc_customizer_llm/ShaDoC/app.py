import ast
import modal
import nodes
from graph_agent import construct_graph
from fastapi import FastAPI, responses
from fastapi.middleware.cors import CORSMiddleware
from lib.common import stub
import lib.utils as utils

web_app = FastAPI(
    title="CodeLangChain Server",
    version="1.0",
    description="Answers questions about TensorFlow API documentation.",
)


# Set all CORS enabled origins
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@stub.function(keep_warm=1,gpu="T4")
@modal.asgi_app()
def serve():
    import logging, os
    logging.disable(logging.WARNING)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    from langchain_core.runnables import RunnableLambda
    from langserve import add_routes

    def inp(input: str) -> dict:
        input_dict =  ast.literal_eval(input)
        try:
            utils.is_valid_api(input_dict['api_name'])
            return {"keys": {
                "question_id": input_dict['question_id'],
                "title": input_dict['title'], 
                "question": input_dict['question'], 
                "api_name": input_dict['api_name'],
                "context": input_dict['context'],
                # "issue_type": input_dict['issue_type'],
                # "ground_truth": input_dict['ground_truth'],
                # "problem": input_dict['problem'],
                # "prompt": input_dict['prompt'], 
                "iterations": 0, "context_iter": 0}}
        except ValueError as e:
            print(e)

    def out(state: dict) -> str:
        if "keys" in state:
            return state["keys"]["response"]
        # elif "generate" in state:
        #     return nodes.extract_response(state=state["generate"])
        # else:
        #     return str(state)

        
    graph = construct_graph().compile()
    chain = RunnableLambda(inp) | graph | RunnableLambda(out)

    add_routes(
        web_app,
        chain,
        path="/codelangchain",
    )

    # redirect the root to the interactive playground
    @web_app.get("/")
    def redirect():
        return responses.RedirectResponse(url="/codelangchain/playground")

    return web_app