import os
import time
import pandas as pd

import lib.global_settings as s
import lib.utils as utils

from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path=dotenv_path)

API_KEY = os.getenv("STACKEXCHANGE_API_KEY")
SO_FILTER = "!0Ryp5rHMvq0Ol-IQ1MwWx-uQy"
url_end = (
    f"key={API_KEY}&order=desc&sort=activity&site=stackoverflow&filter={SO_FILTER}"
)


def user_answers(tags, so_user_ids, chunk_size: int):
    split_list = list(utils.split(so_user_ids, chunk_size))
    responses = []

    for i in range(len(split_list)):
        for k in range(len(split_list[i])):
            url = (
                s.base_url
                + str(split_list[i][k])
                + f"/tags/{tags}/top-answers?{url_end}"
            )
            response = utils.get_response(url)
            responses.append(
                {"user_id": split_list[i][k], "response": response})

        time.sleep(1)

    df_ans = pd.DataFrame(responses)
    df_ans = df_ans.drop_duplicates(subset="user_id", keep="last")
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
    base_url = f"https://api.stackexchange.com/2.3/search/advanced?key={API_KEY}&order=desc&sort=activity&"
    for i in range(len(split_list)):
        for k in range(len(split_list[i])):
            url = (
                base_url
                + f"tagged={tags}&user={str(split_list[i][k])}&site=stackoverflow&filter=!I3ORrkJ*D2Di_GGGhMNc5lC(*QB*Swa6x_ZZUTFPxR-M2HH"
            )
            response = utils.get_response(url)
            responses.append(
                {"user_id": split_list[i][k], "response": response})
        time.sleep(1)

    df_ques = pd.DataFrame(responses)
    df_ques = df_ques.drop_duplicates(subset="user_id", keep="last")
    return df_ques


def so_answers(question_ids, chunk_size: int):
    split_list = list(utils.split(question_ids, chunk_size))
    responses = []
    base_url = "https://api.stackexchange.com/2.3/questions/"
    url_end = f"key={API_KEY}&order=desc&sort=activity&site=stackoverflow&filter=!0Ryp5rMpXhT2LBm9icS-Ar*(m"

    for i in range(len(split_list)):
        for k in range(len(split_list[i])):
            url = base_url + str(split_list[i][k]) + f"/answers?{url_end}"
            response = utils.get_response(url)
            responses.append(
                {"question_id": split_list[i][k], "response": response})

        time.sleep(1)
    df = pd.DataFrame(responses)
    df = df.drop_duplicates(subset="question_id", keep="last")
    return df


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
    response = utils.get_response(url_a)

    q_ids = []
    results = []
    if response["items"]:
        for i in range(len(response["items"])):
            q_ids.append(response["items"][i]["question_id"])

        split_list = list(utils.split(q_ids, 100))
        for x in range(len(split_list)):
            questions_ids = ";".join(map(str, split_list[x]))

            # URL to retrieve the answers based on the question ids (upto 100 question ids separated by ';')
            answer_url = f"questions/{questions_ids}/answers?key={API_KEY}&order=desc&sort=activity&site=stackoverflow&filter=!-KbrbfAqA48jRifMLR7sYu7doHuftBYCT"
            url_b = base_url + answer_url
            answers = utils.get_response(url_b)

            for item in answers["items"]:
                res_dict = {
                    "QuestionId": item["question_id"],
                    "AnswerId": item["answer_id"],
                    "URL": item["link"],
                    "QuestionTitle": item["title"],
                    "Answer": item["body"],
                    "IsAccepted": item["is_accepted"],
                    "CreationDate": item["creation_date"],
                }
                results.append(res_dict)
    return results
