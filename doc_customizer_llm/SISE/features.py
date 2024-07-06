import re
import spacy
import numpy as np
import heapq
from datetime import datetime
from api import StackExchange
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity

from scraper import scraper
from preprocess import tag_code_elements, embed_and_prefix_sentences, pos, split_to_sentences

nlp = spacy.load('en_core_web_md')

class Feature:
    def __init__(self) -> None:
        pass

        
    # Number of token in the sentence
    def compute_tokens(self, sentence, is_tagged=False):
        # nlp = stanza.Pipeline(lang='en', processors='tokenize', tokenize_no_ssplit=True, download_method=None, verbose=False)
        # doc = nlp(sentence)
        doc_tokens = sentence.tokens

        if is_tagged:
            tagged_token_count = 0
            tags = {'code', 'pre', 'a', 'strong', 'em', 'i', 'b', 'h1', 'h2', 'h3', 'sup', 'strike'} #Set for reduced time complexity
            for token in doc_tokens:
                if token.text in tags:
                    tagged_token_count += 1
            return tagged_token_count, len(doc_tokens)
        else:
            return len(doc_tokens)
    

    # def get_api_embeddings(self, text):
    #     nlp = spacy.load('en_core_web_md')
    #     api_sent_embed = []

    #     sentences = self.split_to_sentences(text)
    #     for sent in sentences:
    #         if sent.strip():
    #             api_sent_embed.append(nlp(sent.strip()).vector)
    #     return api_sent_embed
    

    def get_cosine_similarity(self, sentence, api_sent_embed):
        sent_embed = nlp(sentence).vector
        
        similarities = cosine_similarity([sent_embed], api_sent_embed)
        max_similarity_idx = np.argmax(similarities)
        
        max_cosine = similarities[0][max_similarity_idx]
        avg_cosine = np.mean(similarities)
        return max_cosine, avg_cosine
    


    def percentage_of_tokens_tagged(self, sentence):
        tagged_token_count, total_tokens = self.compute_tokens(sentence, is_tagged=True)
        return (tagged_token_count / total_tokens * 100) if total_tokens > 0 else 0
    


    def compute_size(self, text):
        size = 0
        sentences = split_to_sentences(text)
        for sentence in sentences:
            size += self.compute_tokens(sentence)
        return sentences, size



    def count_code_characters(self, sentence):
        soup = BeautifulSoup(str(sentence), 'html.parser')
        code_blocks = soup.find_all(['pre', 'code'])
        code_characters = 0
        seen_blocks = set()

        for block in code_blocks:
            block_text = block.get_text()
            if block_text not in seen_blocks:
                code_characters += len(block_text)
                seen_blocks.add(block_text)
        return code_characters



    def contains_html_tags(self, sentence):
        soup = BeautifulSoup(str(sentence), 'html.parser')
        tags = ['code', 'pre', 'a', 'strong', 'em', 'i', 'b', 'h1', 'h2', 'h3', 'sup', 'strike']
        return bool(soup.find_all(tags))
    


    def starts_with_lowercase(self, sentence):
        if sentence and sentence[0].islower():
            return True
        return False
    


    def is_code_block(self, sentence):
        soup = BeautifulSoup(str(sentence), 'html.parser')
        first_tag = soup.find()
        if first_tag and first_tag.name == 'pre':
            # Check if the content within the <pre> tag is the same as the original sentence
            pre_content = first_tag.decode_contents(formatter="html").strip()
            original_content = sentence.strip()[len("<pre>"):-len("</pre>")].strip()
            return pre_content == original_content
        return False
    


    def find_api_element_position(self, sentence, api_element):
        position = sentence.find(api_element)
        return position if position != -1 else 0
    


    def find_sentence_position(self, answer, sentence):
        position = answer.find(sentence)
        return position
    


    def so_features(self, PackageName, TypeName):
        title_regex = [
            f"(?i).*\b{PackageName}\.{TypeName}\b.*",
            f"(?i).*\b(a |an ){TypeName}\b.*"
        ]       

        body_regex = [
            f".*(^|[a-z]+ |[\.!?] |[\(<]){TypeName}([>\)\.,!?$]| [a-z]+).*",
            f"(?i).*\b{PackageName}\.{TypeName}\b.*",
            f".*<code>.*\b{TypeName}\b.*</code>.*",
            f".*<a.*href.*{PackageName}/{TypeName}\.html.*>.*</a>.*",
        ]

        current_datetime = datetime.now()
        count = 0

        SE = StackExchange()
        response = SE.search(api_name=TypeName)

        api_documentation = scraper(PackageName, TypeName)
        api_documentation = tag_code_elements(api_documentation)
        api_doc_sentences = split_to_sentences(api_documentation)
        api_embeddings, api_documentation = embed_and_prefix_sentences(nlp, api_doc_sentences)

        if response['items']:
            # Questions sorted by relevance
            results = []
            for i, question in enumerate(response['items']):

                if count <= 5:
                    q_title = question['title']
                    q_body = question['body']
                    if any(re.search(r, q_title) for r in title_regex) or any(re.search(r, q_body) for r in body_regex):
                        # contains_api_element = TypeName in q_title or TypeName in q_body
                        ques_score = question.get('score', 0)
                        # ques_favorites = question.get('favorite_count', 0)
                        ques_views = question.get('view_count', 0)
                        # ques_usr_rep = question['owner'].get('reputation', 0)
                        # ques_age = (current_datetime - datetime.fromtimestamp(question['creation_date'])).days
                        ques_ans_count = question.get('answer_count', 0)
                        # ques_usr_accept_rate = question['owner'].get('accept_rate', 0)
                        # is_ques_usr_registered =  question['owner']['user_type'] == "registered"
                        # ques_contains_code = '<code>' in question['body'] or '<pre>' in question['body']

                        if 'last_edit_date' in question:
                            ques_edited = True
                        else:
                            ques_edited = False
                        ques_sentences, ques_size = self.compute_size(question['body'])
                        ques_info = {
                            # "question_contains_api": contains_api_element,
                            "question_score": ques_score,
                            # "question_favorites": ques_favorites,
                            "question_views": ques_views,
                            # "question_user_rep": ques_usr_rep,
                            # "question_age": ques_age,
                            "question_answer_count": ques_ans_count,
                            # "question_user_accept_rate": ques_usr_accept_rate,
                            # "question_user_is_reg": is_ques_usr_registered,
                            # "question_contains_code": ques_contains_code,
                            # "question_edited": ques_edited,
                            "question_size": ques_size,
                        }

                        answers = []
                        # top_10_ans = sorted(question['answers'], key=lambda x: x['score'], reverse=True)[:10]
                        if 'answers' in question and len(question['answers']) > 0:
                            top_10_ans = heapq.nlargest(10, question['answers'], key=lambda x: x['score'])
                            for rank, answer in enumerate(top_10_ans):
                                ans_score = answer.get('score', 0)
                                ans_age = (current_datetime - datetime.fromtimestamp(answer['creation_date'])).days
                                # ans_ques_time_diff = (datetime.fromtimestamp(answer['creation_date']) - datetime.fromtimestamp(question['creation_date'])).days
                                # ans_usr_rep = answer['owner'].get('reputation', 0)

                                # get method for dictionaries allows to provide a default value if a key is not present
                                # ans_usr_accept_rate = answer['owner'].get('accept_rate', 0)

                                # Check the user type is 'registered'
                                # is_ans_usr_registered = answer['owner']['user_type'] == "registered"

                                ans_contains_code = '<code>' in answer['body'] or '<pre>' in answer['body']

                                # if answer['is_accepted'] or 'last_edit_date' in answer:
                                #     ans_accpeted_or_edited = True
                                # else:
                                #     ans_accpeted_or_edited = False

                                answer_rank = rank + 1
                                ans_sentences, ans_size = self.compute_size(answer['body'])
                                ans_info = {
                                    "answer_score": ans_score,
                                    "answer_age": ans_age,
                                    # "answer_question_time_dff": ans_ques_time_diff,
                                    # "answer_user_reputation": ans_usr_rep,
                                    # "answer_user_accept_rate": ans_usr_accept_rate,
                                    # "answer_user_is_reg": is_ans_usr_registered,
                                    "answer_contains_code": ans_contains_code,
                                    # "answer_accepted_or_edit": ans_accpeted_or_edited,
                                    # "answer_rank": answer_rank,
                                    "answer_size": ans_size,
                                }

                                for sent in ans_sentences:
                                    sentence = sent.text.strip()
                                    max_cosine, avg_cosine = self.get_cosine_similarity(sentence, api_embeddings)
                                    sent_info = {
                                        # "sent_codeblock": self.is_code_block(sentence),
                                        "sent_total_tokens": self.compute_tokens(sent),
                                        # "sent_position": self.find_sentence_position(answer['body'], sentence),
                                        # "sent_api_position": self.find_api_element_position(sentence, TypeName),
                                        "sent_start_islower": self.starts_with_lowercase(sentence),
                                        # "sent_code_char_count": self.count_code_characters(sentence),
                                        # "sent_contains_html_tags": self.contains_html_tags(sentence),
                                        # "sent_tagged_token_percent":self.percentage_of_tokens_tagged(sent),
                                        "sent_max_cosine": max_cosine,
                                        "sent_avg_cosine": avg_cosine,
                                        "sentence": sentence,
                                        "sent_pos": pos(sent),
                                    }
                                    results.append({**ques_info, **ans_info, **sent_info})
                else:
                    break
                count += 1
                # Temporary break
                # break
        return results


# F = Feature()
# # sentence = "How to create a new thread in Java?"
# api_documentation = """
# # To create a new thread in Java, you can either implement the Runnable interface or extend the Thread class. The Runnable interface should be implemented by any class whose instances are intended to be executed by a thread. The class must define a method of no arguments called run.
# # Alternatively, a thread can be created by extending the Thread class. This class should override the run method of the superclass.
# # Once a thread object is created, you can start it by calling its start method, which causes the thread to begin execution; the Java Virtual Machine calls the run method of the thread.
# # """
# # sentence = "Here is the target sentence."
# result = F.compute_size(api_documentation)
# print(result)