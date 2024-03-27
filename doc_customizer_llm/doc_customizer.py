# ==========================================================================================================================
# IMPORT DEPENDENCIES
# ==========================================================================================================================
import os
import ast
from langchain.chains.prompt_selector import ConditionalPromptSelector

COLOR = {
    "HEADER": "\033[95m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "RED": "\033[91m",
    "ENDC": "\033[0m",
}

# Custom modules
from lib.utils import remove_broken_urls, create_markdown_file, update_cell
from lib.helper_funcs import prompt_task, read_markdown_file

import templates.template_01 as template_01
import templates.template_02 as template_02

from components.AIScaper.main_scaper import ai_webscraper
from components.AIWebSearch.search import course_urls_retriever
from components.AIWebSearch.cohere_rag_retriever import relevant_context_retriever
from components.SORetriever.so_retrieval import retrieve_relevant_from_so
from components.Customizer.main_customizer import ai_doc_customizer
from components.BinaryClassifier.main_binaryclassifier import question_classifier
from components.QuestionSummarizer.summarize import question_summarizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")   # LangChain LLM API Key (To use LangSmith)

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "langchain-doc-customizer"

# ==========================================================================================================================
# MAIN CODE
# ==========================================================================================================================
issue_types_require_examples = [
    "Documentation Replication on Other Examples", 
    "Documentation Replicability", 
    "Inadequate Examples"]

description_only_issue_types = [
    "Documentation Ambiguity",
    "Documentation Completeness"]

def doc_customization(df, row_index):
    flag = 0
    if df['QuestionAPI'][row_index].startswith("tf."):
        # Read markdown file of the corresponding API from the database directory
        documentation = read_markdown_file(df['QuestionAPI'][row_index])
        if documentation is not None:
            # doc_related = question_classifier(df['Question'][row_index])
            doc_related = 1 #Manually initialized just for testings
            if doc_related == 1:
                # print("Doc-related Question")
                definition, task = prompt_task(df['IssueType'][row_index])

                # Question summarization and intent identification
                intent = question_summarizer(df['Title'][row_index], df['Question'][row_index])
                update_cell(df, 'Intent', row_index, str(intent))

                # Search web
                search_results = course_urls_retriever(intent)
                course_urls = search_results['urls']
                if not course_urls:
                    course_urls = remove_broken_urls(course_urls)

                # Retrieve relevant context and sources using Cohere RAG Retriever (Web Connector)
                model_response, context, urls = relevant_context_retriever(intent)
                context = "\n".join(context)
                update_cell(df, 'Context', row_index, [context])
               

                if df['IssueType'][row_index] in issue_types_require_examples:
                    relevant_so_answers = retrieve_relevant_from_so(df['Title'][row_index], df['Question'][row_index])

                elif df['IssueType'][row_index] == "Requesting (Additional) Resources":
                    flag = 1
                    relevant_so_answers = retrieve_relevant_from_so(df['Title'][row_index], df['Question'][row_index])

                
                if flag == 0:
                    PROMPT, prompt = template_01.return_prompt(
                        context, 
                        df['Title'][row_index], 
                        df['Question'][row_index], 
                        df['IssueType'][row_index], 
                        definition, 
                        df['QuestionAPI'][row_index], 
                        documentation, task, relevant_so_answers, urls, course_urls)
                elif flag == 1:
                    template, prompt = template_02.return_prompt()
                
                customization = ai_doc_customizer(PROMPT, prompt)
                customization = customization[0]['text'] 
                update_cell(df, 'Answer', row_index, str(customization))
            else:
                print(f"{COLOR['RED']}📦: QUESTION IS NOT DOCUMENTATION RELATED.{COLOR['ENDC']}\n")
        else:
            print(f"{COLOR['RED']}📦: QUESTION IS NOT DOCUMENTATION RELATED.{COLOR['ENDC']}\n")
    return df

# ==========================================================================================================================
# TEST EXECUTIONS
# ==========================================================================================================================


# title = "TF1 to TF2 migration"
# body = """"
#     <p>Hello I am new to tensorflow and I am working on a code that I would like to migrate from tensorflow 1 to 2. I have this line of code:</p>
#     <pre><code>x1 = tf.compat.v1.placeholder(tf.float32, [], name=&quot;x1&quot;)
#     </code></pre>
#     <p>As mentioned in <a href="https://www.tensorflow.org/api_docs/python/tf/compat/v1/placeholder" rel="nofollow noreferrer">https://www.tensorflow.org/api_docs/python/tf/compat/v1/placeholder</a>, I should use <code>keras.Input</code>. But even when specifying the shape, I can't have the same tensor as with compat.v1:</p>
#     <pre><code>x2 = tf.keras.Input(shape=[], dtype=tf.float32, name=&quot;x2&quot;)
#     </code></pre>
#     <p>To check the shape I use <code>tf.shape(x1)</code> or <code>tf.shape(x2)</code>, but the shapes are not the same. Could anyone explain to me how to have, in TF2, the same shape as in TF1 ?
#     Thanks and regards</p>
# """
# api_name = "tf.compat.v1.placeholder"

# customization = doc_customization(title, body, api_name)
# create_markdown_file(customization, "customized.md")