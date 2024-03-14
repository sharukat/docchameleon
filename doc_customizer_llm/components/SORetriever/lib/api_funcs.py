# ==========================================================================================================================
# IMPORT DEPENDENCIES
# ==========================================================================================================================

import os
import sys
import json
import requests
from dotenv import load_dotenv

# Load custom modules
import lib.utils as util

# ==========================================================================================================================
# LOAD API KEYS FROM THE .env FILE
# ==========================================================================================================================

load_dotenv(dotenv_path="/Users/sharukat/Documents/ResearchYU/Code/doc-customizer-llm/doc_customizer_llm/.env")
API_KEY = os.getenv("STACKEXCHANGE_API_KEY")   # Claude LLM API Key


# ==========================================================================================================================
# MAIN CODE
# ==========================================================================================================================

def query_so(query: str):
    """
    The query_so function takes in a query and returns the response from the stackexchange api

    Args:
        query: str: The query to be passed to the stackexchange api

    Returns:
        A dataframe with the response
    """
    
    base_url = "https://api.stackexchange.com/2.3/"

    # URL to retrieve the response from the stackexchange api based on the search query
    search_url = f"search/excerpts?key={API_KEY}&order=desc&sort=activity&q={query}&accepted=True&answers=1&tagged=TensorFlow;&site=stackoverflow&filter=!3tlXYAfMSMBBM)Mj)"
    url_a = base_url + search_url
    response = requests.get(url_a)
    response = json.loads(response.text)

    q_ids = []
    results = []
    if response['items']:
        for i in range(len(response['items'])):
            q_ids.append(response['items'][i]['question_id'])

        split_list = list(util.split(q_ids, 100))
        for x in range(len(split_list)):
            questions_ids = ';'.join(map(str, split_list[x]))

            #URL to retrieve the answers based on the question ids (upto 100 question ids separated by ';')
            answer_url =f"questions/{questions_ids}/answers?key={API_KEY}&order=desc&sort=activity&site=stackoverflow&filter=!-KbrbfAqA48jRifMLR7sYu7doHuftBYCT"
            url_b = base_url + answer_url
            answers = requests.get(url_b)
            answers = json.loads(answers.text)

            for item in answers['items']:
                res_dict = {
                    "QuestionId": item['question_id'],
                    "AnswerId": item['answer_id'],
                    "URL": item['link'],
                    "QuestionTitle": item['title'],
                    "Answer": item['body'],
                    "IsAccepted": item['is_accepted'],
                    "CreationDate": item['creation_date']

                }
                results.append(res_dict)
    return results