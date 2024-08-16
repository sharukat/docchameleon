import os
from typing import List, Dict, TypedDict
from lib.config import DOCUMENTATION_PATH

import modal

image = modal.Image.debian_slim(python_version="3.11").from_registry(
        "tensorflow/tensorflow:latest-gpu"
    ).env(
        {"TF_ENABLE_ONEDNN_OPTS": '0', "TF_CPP_MIN_LOG_LEVEL": "2"}
    ).pip_install(
        "human_eval==1.0.3",
        "nltk==3.8.1",
        "protobuf==3.20.*",
        "tiktoken==0.5.2",
        "langchain==0.1.19",
        "langgraph==0.0.30",
        "langchain-community==0.0.38",
        "langchain-openai==0.0.5",
        "langserve[all]==0.2.1",
        "langchain-voyageai==0.1.0",
        "cohere==5.3",
        "chromadb==0.4.24",
        "langchainhub==0.1.15",
        "mdutils==1.6.0",
        "numpy==1.26.4",
        "pandas==2.2.1",
        "ragas==0.1.4",
        "matplotlib==3.9.2",
        "scikit-learn==1.5.1",
        # "setfit==1.0.3",
        "langchain-experimental==0.0.58",
        "langchain-cohere==0.1.4",
    )


stub = modal.Stub(
    "code-langchain",
    image=image,
    # volumes={"/my-vol":modal.Volume.from_name("my-docs-volume")},
    mounts=[modal.Mount.from_local_dir(DOCUMENTATION_PATH, remote_path="/root/docs")],
    secrets=[
        modal.Secret.from_name("my-openai-secret"),
        modal.Secret.from_name("my-langsmith-secret"),
        modal.Secret.from_name("my-cohere-secret"),
        modal.Secret.from_name("my-voyage-secret"),
        modal.Secret.from_name("my-stackexchange-secret"),
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


class RagGraphState(TypedDict):
    """|
    Represents the state of our graph.
    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """
    question : str
    generation : str
    documents : List[str]
    hallucinations: str
    answer_relevancy: str
    iterations: int