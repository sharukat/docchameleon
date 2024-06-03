# ==========================================================================================================================
# IMPORT DEPENDENCIES
# ==========================================================================================================================

import json
import requests
from dotenv import load_dotenv

# Load custom modules
from lib.utils import split

API_KEY = "BmopG%29d9Thccirg4e%29CjOw%28%28"


# ==========================================================================================================================
# MAIN CODE
# ==========================================================================================================================

class StackExchange:
    def __init__(self) -> None:
        self.base_url = "https://api.stackexchange.com/2.3/"

    def get_response(self, url) -> dict:
        full_url = self.base_url + url
        response = requests.get(full_url)
        response = json.loads(response.text)
        return response

    def answers_using_qids(self, question_ids) -> dict:
        #URL to retrieve the answers based on the question ids (upto 100 question ids separated by ';')
        answer_url =f"questions/{question_ids}/answers?key={API_KEY}&order=desc&sort=activity&site=stackoverflow&filter=!-KbrbfAqA48jRifMLR7sYu7doHuftBYCT"
        response = self.get_response(url=answer_url)
        return response
    

    def similar_qids_based_on_title(self, title) -> dict:
        # URL to retrieve the similar questions based on the title
        search_url = f"similar?key={API_KEY}&order=desc&sort=relevance&tagged=tensorflow&title={title}&site=stackoverflow&filter=!.FdHWktosn0Qhc2uAM-Sixm1hmLQS"
        response = self.get_response(url=search_url)
        return response

    def related_questions_based_on_id(self, qid) -> dict:
        questions_url = f"questions/{qid}/related?key={API_KEY}&order=desc&sort=activity&site=stackoverflow&filter=!WWt6kbMxuVVOPrYviQ978rZ7VGv-AZ12PEwS3xD"
        response = self.get_response(url=questions_url)
        return response



def stackexchange(query: str):
    """
    The query_so function takes in a query and returns the response from the stackexchange api
    Args:
        query: str: The query to be passed to the stackexchange api
    Returns:
        A dataframe with the response
    """

    SE = StackExchange()

    # URL to retrieve the similar questions based on the title
    response = SE.similar_qids_based_on_title(query)

    q_ids = []
    results = []
    if response['items']:
        for i in range(len(response['items'])):
            if response['items'][i]['answer_count'] > 0:
                q_ids.append(response['items'][i]['question_id'])
        split_list = list(split(q_ids, 100))
        for x in range(len(split_list)):
            questions_ids = ';'.join(map(str, split_list[x]))

            #URL to retrieve the answers based on the question ids (upto 100 question ids separated by ';')
            answers = SE.answers_using_qids(questions_ids)
            for item in answers['items']:
                if item['is_accepted'] == True:
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
                else:
                    pass
    return results




# Test
# title = "tf.keras.metrics.SpecificityAtSensitivity num_thresholds interpretation"
# results = stackexchange(title)
# print(results)