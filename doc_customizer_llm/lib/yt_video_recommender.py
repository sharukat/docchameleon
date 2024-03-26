from youtube_transcript_api import YouTubeTranscriptApi
import json
from langchain.docstore.document import Document
from langchain.embeddings.fastembed import FastEmbedEmbeddings
from langchain.retrievers import ParentDocumentRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain_community.embeddings import CohereEmbeddings
import uuid
from langchain_community.document_loaders import YoutubeLoader
import pandas as pd
from googleapiclient.discovery import build
from datetime import datetime, timedelta
import chromadb.utils.embedding_functions as embedding_functions
from langchain_community.llms import Cohere

from dotenv import load_dotenv
import os
import cohere

load_dotenv()

co = cohere.Client(os.getenv('COHERE_API_KEY'))
TF_channel_id = 'UC0rqucBdTuFTjJiefW5t-IQ'

def consider_video(video_date_str):
    TF_api_last_updated = datetime.strptime('2023-05-26 00:00:00', '%Y-%m-%d %H:%M:%S')

    video_date = datetime.strptime(video_date_str, '%Y-%m-%d %H:%M:%S')
    difference = video_date - TF_api_last_updated

    return abs(difference) < timedelta(days=365 * 6)

def get_transcripts(youtube_videos):
    '''
    Parameters:
        youtube_videos (list): Youtube video links

    Return: 
        documents (list): String transcript for each video link
        transcript_jsons(list): Json transcript for each video link
    '''
    documents = []
    transcript_jsons = []
    index = 0
    for i in range(len(youtube_videos)):
        try:
            video_id = youtube_videos[i].split('https://www.youtube.com/watch?v=')[1]
            transcript_json = YouTubeTranscriptApi.get_transcript(video_id)
            # print(transcript_json)
            transcript_text = ' '.join([snippet['text'] for snippet in transcript_json])
            doc = Document(page_content = transcript_text, metadata ={"video_id": video_id, 'index': index})
            documents.append(doc)
            transcript_jsons.append(transcript_json)
            index += 1

        except:
            pass
            # print('error generating transcript for index, video:', (i, youtube_videos[i]))

    return documents, transcript_jsons

def get_channel_videos(query):
    '''
    Parameters:
        query (str): Query to search Youtube content

    Return:
        videos (list): recommended videos for a given query
    '''
    youtube = build('youtube', 'v3', developerKey=os.getenv('YOUTUBE_API_KEY'))

    # Get videos from channel
    videos = []

    playlist_response = youtube.search().list(
        q=query,
        part='snippet',
        maxResults=100,
        order='relevance',
        type='video',
    ).execute()

    for item in playlist_response['items']:
        video_id = item['id']['videoId']
        video_link = f'https://www.youtube.com/watch?v={video_id}'
        videos.append(video_link)

    return videos


def find_start_time(doc, transcript):
    '''
    We find the starting point where the first 5 words of doc match within the entire transcript. Since doc is a continuous substring of the entire transcript, we are guaranteed
    to find a starting time for the given doc.
    Each text in transcipt contains only a few words, hence it might not contain all first 5 words of doc. So, we also check the next 'text' of transcript.

    Parameters:
        doc (Doc): Recommended parent document based on similarity search
        transcript (Json): Json transcript of the recommended video for doc
    
    Return:
        start (int): Start time for the recommended video
    '''
    doc_words = doc.page_content.split(' ')
    first_5_words = doc_words[:5]
    
    for i in range(len(transcript) - 1):
        start = transcript[i]['start']
        flag = True

        for word in first_5_words:
            if word not in transcript[i]['text'] and word not in transcript[i+1]['text']:
                flag = False
                break
        
        if flag:
            return start

    return None

def relevant_score_videos(question_title, relevant_docs, transcript_jsons):
    '''
    Need to suggest youtube videos based on relevant_docs. If a doc has a similarity score > 0.5, then we want to print out it's relevance_score, 
    recommended video and the start time for that video.

    Parameters:
        question_title (str): Question title to search recommended videos for
        relevant_docs (list): List of relevant Docs based on similarity search and Cohere ReRank
        transcript_jsons (list): Full Json transcripts based on recommended doc
    
    Return:
        None
    '''
    relevant_videos_start_time = {}

    for doc in relevant_docs:
        relevance_score = doc.metadata['relevance_score']

        if relevance_score > 0.5:
            transcript = transcript_jsons[doc.metadata['index']]
            start_time = find_start_time(doc, transcript)

            relevant_videos_start_time[doc.metadata['video_id']] = [relevance_score, start_time]

    if len(relevant_videos_start_time):
        print('\nRecommended videos for question title: ', question_title)
        for key, value in relevant_videos_start_time.items():

            print(f'\nRelevance Score: {value[0]}')
            print(f'https://www.youtube.com/watch?v={key} at time = {value[1]}')
    else:
        print('No recommended videos for question title: ', question_title)


current_dir = os.path.dirname(__file__)
relative_path = '../data/DocQues_AcceptedAns_Issues_v1.csv'
csv_file_path = os.path.normpath(os.path.join(current_dir, relative_path))

df = pd.read_csv(csv_file_path)
question_body = list(df['Body'])
question_title = list(df['Title'])

cohere_ef  = CohereEmbeddings(model="embed-english-light-v3.0")

vectorstore = Chroma(
  collection_name=f"split_parents{str(uuid.uuid4())}",
  embedding_function=FastEmbedEmbeddings(),
  persist_directory="./chroma_db"
)
store = InMemoryStore()

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

parent_retriever = ParentDocumentRetriever(
  vectorstore=vectorstore,
  docstore=store,
  child_splitter=child_splitter,
  parent_splitter=parent_splitter,
)
child_retriever = vectorstore.as_retriever()

compressor = CohereRerank()
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=parent_retriever)


for i in range(len(question_title)):

    recommended_videos = get_channel_videos(question_title[i])
    generated_transcripts, transcript_jsons = get_transcripts(recommended_videos)
    parent_retriever.add_documents(generated_transcripts)

    compressed_docs = compression_retriever.get_relevant_documents(query=question_body[i])
    # print(compressed_docs)
    relevant_score_videos(question_title[i], compressed_docs, transcript_jsons)
    
    delete_ids = vectorstore.get()['ids']
    vectorstore.delete(ids=delete_ids)