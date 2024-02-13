import time
import pandas as pd
import global_settings as s
from urllib.parse import urljoin
import utils as utils

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path=dotenv_path)

import os
API_KEY = os.getenv("STACKEXCHANGE_API_KEY")


def user_answers(tags, so_user_ids, chunk_size: int):
    split_list = list(utils.split(so_user_ids, chunk_size))
    responses = []
    url_end = f'key={API_KEY}&order=desc&sort=activity&site=stackoverflow&filter=!0Ryp5rHMvq0Ol-IQ1MwWx-uQy'

    for i in range(len(split_list)):
        for k in range(len(split_list[i])):
            url = s.base_url + str(split_list[i][k]) + f'/tags/{tags}/top-answers?{url_end}'
            response = utils.get_response(url)
            responses.append({'user_id': split_list[i][k], 'response': response})

        time.sleep(1)

    df_ans = pd.DataFrame(responses)
    df_ans = df_ans.drop_duplicates(subset='user_id', keep='last')
    return df_ans



def user_questions(tags, so_user_ids, chunk_size: int):
    """
    The user_questions function takes in a list of tags, a list of user ids, and an optional chunk size.
    It then returns the questions for each user id that is associated with the given tag(s).

    Args:
        tags: Specify the tags for which you want to get answers
        so_user_ids: Pass the list of user ids to be used in the api call
        chunk_size: int: Define the number of user ids to be passed in each request

    Returns:
        A dataframe with the user_id and response
    """
    split_list = list(utils.split(so_user_ids, chunk_size))
    responses = []
    base_url =f'https://api.stackexchange.com/2.3/search/advanced?key={API_KEY}&order=desc&sort=activity&'
    for i in range(len(split_list)):
        for k in range(len(split_list[i])):
            url = base_url + f'tagged={tags}&user={str(split_list[i][k])}&site=stackoverflow&filter=!I3ORrkJ*D2Di_GGGhMNc5lC(*QB*Swa6x_ZZUTFPxR-M2HH'
            response = utils.get_response(url)
            responses.append({'user_id': split_list[i][k], 'response': response})
        time.sleep(1)

    df_ques = pd.DataFrame(responses)
    df_ques = df_ques.drop_duplicates(subset='user_id', keep='last')
    return df_ques


def so_answers(question_ids, chunk_size: int):
    split_list = list(utils.split(question_ids, chunk_size))
    responses = []
    base_url = 'https://api.stackexchange.com/2.3/questions/'
    url_end = f'key={API_KEY}&order=desc&sort=activity&site=stackoverflow&filter=!0Ryp5rMpXhT2LBm9icS-Ar*(m'

    for i in range(len(split_list)):
        for k in range(len(split_list[i])):
            url = base_url + str(split_list[i][k]) + f'/answers?{url_end}'
            response = utils.get_response(url)
            responses.append({'question_id': split_list[i][k], 'response': response})

        time.sleep(1)
    df = pd.DataFrame(responses)
    df = df.drop_duplicates(subset='question_id', keep='last')
    return df