import os
import json

def prompt_task(type: str):
  
    # llm_tasks = {
    #     "Documentation Replication on Other Examples": {
    #     "definition": "Issues related to replicating the documentation examples on other different examples.", 
    #     "task": """Your task is to generate complete executable code example with an explanation replicating the corresponding example 
    #     in the documentation to address the user's quesetion body by only using the knowledge provided as 'context'."""
    #     },
    #     "Documentation Ambiguity": {
    #     "definition": "Issues related to not understanding the content properly.", 
    #     "task": """Provide step by step explanations and examples to understand the content of the documentation with respect to the 
    #     question asked by the user."""
    #     },
    #     "Documentation Completeness": {
    #     "definition": "Issues that mention that the documentation is incomplete or missing information.", 
    #     "task": "Complete the documentation by adding the information with respect to the question asked by the user."
    #     },
    #     "Documentation Replicability": {
    #     "definition": "Issues related to replicating the documentation examples.", 
    #     "task": """Your task is to generate complete executable code example with an explanation to address the user's quesetion body 
    #     by only using the knowledge provided as 'context'."""
    #     },
    #     "Inadequate Examples": {
    #     "definition": "Issues that mention the documentation has insufficient examples.", 
    #     "task": "Provide multiple complete examples to showcase its usability with respect to the question asked by the user."
    #     },
    #     "Lack of Alternative Solutions": {
    #     "definition": "Issues that mention the unavailability of alternative solutions or documentation.", 
    #     "task": "Provide alternative solutions with respect to the question asked by the user."
    #     },
    #     "Requesting (Additional) Documentation/ Examples": {
    #     "definition": "Questions that request additional examples or documentations as a support.", 
    #     "task": "Provide links to reliable additional documentation or examples with respect to the question asked by the user."
    #     }
    # }

    llm_tasks = {
        "examples_required": {
        "definition": "Answer to the question should contain an example that includes an explanation and the code.", 
        "task": """Your task is to generate complete executable code example with an explanation to address the user's quesetion body 
        by only using the knowledge provided as 'context'."""
        },

        "description_only": {
        "definition": "Answer to the question should only contain an explanation.", 
        "task": """Your task is to generate an explanation only to address the user's quesetion body 
        by only using the knowledge provided as 'context'."""
        },
    }
    return llm_tasks.get(type)