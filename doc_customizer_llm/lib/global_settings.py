import os

ROOT = "/Users/sharukat/Documents/ResearchYU/Code/doc-customizer-llm"
DOCUMENTATION_PATH = os.path.join(ROOT, "doc_customizer_llm/tf_api_docs")
DATA_PATH = "./data"
VECTORDB_PATH = "./vectordb"
ASSETS_PATH = os.path.join(ROOT, "doc_customizer_llm/assets")
DOTENV_PATH = os.path.join(ROOT, "doc_customizer_llm/.env")

COLOR = {
    "HEADER": "\033[95m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "RED": "\033[91m",
    "ENDC": "\033[0m",
}

# Stack Exchange APIs
base_url ='https://api.stackexchange.com/2.3/users/'