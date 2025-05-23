from components.QuestionSummarizer.summarize import question_summarizer
from components.Customizer.main_customizer import ai_doc_customizer
from components.SORetriever.so_retrieval import retrieve_relevant_from_so
from components.AIWebSearch.cohere_rag_retriever import relevant_context_retriever
from components.AIWebSearch.search import course_urls_retriever
import templates.template_02 as template_02
import templates.template_01 as template_01
from lib.helper_funcs import prompt_task, read_markdown_file
from lib.utils import remove_broken_urls, update_cell
import os

COLOR = {
    "HEADER": "\033[95m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "RED": "\033[91m",
    "ENDC": "\033[0m",
}

os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["LANGCHAIN_API_KEY"] = os.getenv(
    "LANGCHAIN_API_KEY"
)  # LangChain LLM API Key (To use LangSmith)

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "langchain-doc-customizer"

# ==========================================================================================================================
# MAIN CODE
# ==========================================================================================================================
issue_types_require_examples = [
    "Documentation Replication on Other Examples",
    "Documentation Replicability",
    "Inadequate Examples",
]

description_only_issue_types = [
    "Documentation Ambiguity", "Documentation Completeness"]


def doc_customization(df, row_index):
    flag = 0
    if df["QuestionAPI"][row_index].startswith("tf."):
        # Read markdown file of the corresponding API from the database directory
        documentation = read_markdown_file(df["QuestionAPI"][row_index])
        if documentation is not None:
            doc_related = 1  # Manually initialized just for testings
            if doc_related == 1:
                definition, task = prompt_task(df["IssueType"][row_index])

                # Question summarization and intent identification
                intent = question_summarizer(
                    df["Title"][row_index], df["Question"][row_index]
                )
                update_cell(df, "Intent", row_index, str(intent))

                # Search web
                search_results = course_urls_retriever(intent)
                course_urls = search_results["urls"]
                if not course_urls:
                    course_urls = remove_broken_urls(course_urls)

                # Retrieve relevant context and sources using Cohere RAG Retriever (Web Connector)
                _, context, urls = relevant_context_retriever(intent)
                context = "\n".join(context)
                update_cell(df, "Context", row_index, context)

                relevant_so_answers = None
                if df["IssueType"][row_index] in issue_types_require_examples:
                    relevant_so_answers = retrieve_relevant_from_so(
                        df["Title"][row_index], df["Question"][row_index]
                    )

                elif df["IssueType"][row_index] == "Requesting (Additional) Resources":
                    flag = 1
                    relevant_so_answers = retrieve_relevant_from_so(
                        df["Title"][row_index], df["Question"][row_index]
                    )

                if flag == 0:
                    PROMPT, prompt = template_01.return_prompt(
                        context,
                        df["Title"][row_index],
                        df["Question"][row_index],
                        df["IssueType"][row_index],
                        definition,
                        df["QuestionAPI"][row_index],
                        documentation,
                        task,
                        relevant_so_answers,
                        urls,
                        course_urls,
                    )
                elif flag == 1:
                    _, prompt = template_02.return_prompt()

                customization = ai_doc_customizer(PROMPT, prompt)
                customization = customization[0]["text"]
                update_cell(df, "Answer", row_index, str(customization))
            else:
                print(
                    f"{COLOR['RED']}📦: QUESTION IS NOT DOCUMENTATION RELATED.{COLOR['ENDC']}\n"
                )
        else:
            print(
                f"{COLOR['RED']}📦: QUESTION IS NOT DOCUMENTATION RELATED.{COLOR['ENDC']}\n"
            )
    return df
