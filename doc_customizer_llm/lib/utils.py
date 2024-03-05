from pathlib import Path
import glob
import json
import requests
import ast
import numpy as np
import pandas as pd
import lib.global_settings as s

from statistics import mean
from datetime import datetime

def make_user_dataset(df, type='questions'):

    user_dataset = []
    for index, row in df.iterrows():
        user_id = row['user_id']
        response = row['response']
        response = ast.literal_eval(response)

        up_vote_count   = []
        down_vote_count = []
        count = 0
        reputation = 0
        if response['items']:
            for item in response['items']:
                up_vote_count.append(item['up_vote_count'])
                down_vote_count.append(item['down_vote_count'])
                if type == 'answers':
                    if item['is_accepted']:
                        count = count + 1

            total_up_votes = sum(up_vote_count)
            total_down_votes = sum(down_vote_count)
            if type == 'answers':
                reputation = (total_up_votes * 10) + (count * 15) - (total_down_votes * 2)
                v_index = sum(x >= n+1 for n, x in enumerate(sorted(  list(up_vote_count), reverse=True)))
                user_dataset.append({
                    'user_id': user_id, 'v_index': v_index, 'reputation': reputation})

            else:
                reputation = (total_up_votes * 10) - (total_down_votes * 2)
                user_dataset.append({
                    'user_id': user_id, 'reputation': reputation})

        else:
            if type == 'answers':
                user_dataset.append({
                    'user_id': user_id, 'v_index': 0, 'reputation':0 })
            else:
                user_dataset.append({
                    'user_id': user_id, 'reputation': 0})
            
    dataset = pd.DataFrame(user_dataset)
    return dataset




def get_accepted_answer(df):
    pass



# --------------------------------------------------------------------------------------------------------------------------------------
def get_first_date(df):
    """
    The get_first_date function takes a dataframe as input and returns a new dataframe with the user_id and first question date.
        The function iterates through each row of the input dataframe, extracts the user_id and response from that row,
        converts response to dictionary format using ast.literal_eval(), checks if there are any items in response['items'],
        creates a list comprehension of all question timestamps for that user (if there are any), finds minimum timestamp value,
        appends this information to an empty list called first_so_question which is then converted into a dataframe

    Args:
        df: Pass in the dataframe that we want to use

    Returns:
        A dataframe with the first question date for each user
    """

    first_so_question = []
    for index, row in df.iterrows():
        user_id = row['user_id']
        response = row['response']
        response = ast.literal_eval(response)
        if response['items']:
            # List comprehension
            question_timestamps = [datetime.fromtimestamp(item['creation_date']).date() for item in response['items']]    
            first_so_question.append({'user_id': user_id, 'first_q_date': min(question_timestamps)})
        else:
            first_so_question.append({'user_id': user_id, 'first_q_date': None})
            
    df_first_so_question = pd.DataFrame(first_so_question)
    df_first_so_question = df_first_so_question.drop_duplicates(subset='user_id', keep='last')

    return df_first_so_question

# --------------------------------------------------------------------------------------------------------------------------------
# compute user experience based on tags
def compute_experience(df, relative_exp=False, in_months=False):
    """
    The compute_experience function takes in a dataframe and returns the experience of each user.
        The function also takes in two optional parameters: relative_exp and in_months.

        If relative_exp is set to True, then the experience will be calculated from the first question asked by
            that user until last question date. Otherwise, it will be calculated from their first question
            until current date.

    Args:
        df: Pass the dataframe to the function
        relative_exp: Determine whether the experience is relative to the last question date or not
        in_months: Determine whether the experience is in years or months

    Returns:
        A dataframe with the user_id and experience
    """

    date_diff = []
    for index, row in df.iterrows():
        user_id = row['user_id']
        start_date = row['first_q_date']
        end_date = datetime.strptime(row['creation_date'], '%Y-%m-%d').date() if relative_exp == True else datetime.now()
        if start_date is not None:
            if in_months:
                diff = (end_date.year - start_date.year)*12 + (end_date.month -  start_date.month)
            else:
                diff = end_date.year - start_date.year
            date_diff.append({'user_id': user_id, 'experience': diff})
        else:
            date_diff.append({'user_id': user_id, 'experience': 0})
    return pd.DataFrame(date_diff)


def split(list_a, chunk_size):
    """
    The split function takes a list and splits it into chunks of the specified size.

    Args:
        list_a: Specify the list that will be split into chunks
        chunk_size: Specify the size of each chunk

    Returns:
        A generator object
    """
    for i in range(0, len(list_a), chunk_size):
        yield list_a[i:i + chunk_size]


def get_response(url):
    """
    The get_response function takes an url as an argument and returns the response from that url in json format.

    Args:
        url: Specify the url of the api call

    Returns:
        A dictionary
    """
    response = requests.get(url)
    response = json.loads(response.text)
    return response



def processed_dataset(df, type='questions'):
    user_dataset = [
        {
            'user_id': row['user_id'],
            **process_response(ast.literal_eval(row['response']), type)
        }
        for _, row in df.iterrows()
    ]
    dataset = pd.DataFrame(user_dataset)
    return dataset


def process_response(response, type):
    upvote_counts = [item['up_vote_count'] for item in response['items']]
    down_vote_counts = [item['down_vote_count'] for item in response['items']]
    total_up_votes = sum(upvote_counts)
    total_down_votes = sum(down_vote_counts)
    
    if type == 'answers':
        accepted_ans_count = sum(item['is_accepted'] for item in response['items'])
        v_index = sum(x >= n+1 for n, x in enumerate(sorted(upvote_counts, reverse=True)))
        reputation = (total_up_votes * 10) + (accepted_ans_count * 15) - (total_down_votes * 2)
        return {
            'v_index': v_index,
            'reputation': reputation
        }
    else:
        reputation = (total_up_votes * 10) - (total_down_votes * 2)
        return {'reputation': reputation}

