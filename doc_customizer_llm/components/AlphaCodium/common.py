import os
from typing import Dict, TypedDict

import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
        "tensorflow-cpu",
        "protobuf==3.20.*",
        "tiktoken==0.5.2",
        "langchain==0.1.11",
        "langgraph==0.0.28",
        "langchain_community==0.0.25",
        "langchain-openai==0.0.5",
        "langserve[all]==0.0.51",
        "cohere==4.52",
        "chromadb==0.4.24",
        "langchainhub==0.1.15",
        "numpy",
        "pandas",
    )


stub = modal.Stub(
    "code-langchain",
    image=image,
    secrets=[
        modal.Secret.from_name("my-openai-secret"),
        modal.Secret.from_name("my-langchain-secret"),
        modal.Secret.from_name("my-cohere-secret"),
        modal.Secret.from_name("my-tf-secret"),
    ],
)


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """

    keys: Dict[str, any]


os.environ["LANGCHAIN_PROJECT"] = "codelangchain"

COLOR = {
    "HEADER": "\033[95m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "RED": "\033[91m",
    "ENDC": "\033[0m",
}