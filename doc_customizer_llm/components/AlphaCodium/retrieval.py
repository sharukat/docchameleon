# ==========================================================================================================================
# IMPORT DEPENDENCIES
# ==========================================================================================================================

# LANGCHAIN MODULES
from langchain_community.chat_models import ChatCohere
from langchain.retrievers.document_compressors import CohereRerank
from langchain_community.retrievers import CohereRagRetriever
from langchain.retrievers import ContextualCompressionRetriever


# ==========================================================================================================================
# MAIN CODE
# ==========================================================================================================================

def relevant_context_retriever(query: str):
    print("Retrieving relevant context from web using Cohere RAG Retriever....")
    urls = []
    docs = []
    model_response = ""
    rag = CohereRagRetriever(llm=ChatCohere(model="command-r"), connectors=[{"id": "web-search"}]) 

    compressor = CohereRerank(model='rerank-english-v2.0', top_n=5)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=rag , 
    )
    
    # docs = rag.get_relevant_documents(query)
    reranked_docs = compression_retriever.get_relevant_documents(query=query)
    for index, doc in enumerate(reranked_docs):
        if 'type' in doc.metadata:
            model_response = doc.page_content
        else:
            if doc.metadata['relevance_score'] > 0.6:
                urls.append(doc.metadata['url'])
                docs.append(doc)
    # for doc in docs[:-1]:
    #     urls.append(doc.metadata['url'])
    #     page_content.append(doc.page_content)

    print("Completed Successfully\n")
    return model_response
