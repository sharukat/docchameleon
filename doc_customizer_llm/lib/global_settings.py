import os

ROOT = "/Users/sharukat/Documents/ResearchYU/Code/doc-customizer-llm"
DOCUMENTATION_PATH = os.path.join(ROOT, "doc_customizer_llm/tf_api_docs")
DATA_PATH = "./data"
VECTORDB_PATH = "./vectordb"
ASSETS_PATH = os.path.join(ROOT, "doc_customizer_llm/assets")
DOTENV_PATH = os.path.join(ROOT, "doc_customizer_llm/.env")

# Stack Exchange APIs
base_url ='https://api.stackexchange.com/2.3/users/'