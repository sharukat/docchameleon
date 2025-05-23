from langchain.prompts import PromptTemplate


def return_prompt(
    context,
    title,
    body,
    issue_type,
    definition,
    api_name,
    documentation,
    task,
    qa,
    urls,
    course_urls,
):
    template = """
        Your task is to provide a customized response to a user's question based on the given context, documentation, and the task itself. 
        Here are the steps to follow:

        First, review the context provided in {context}. This context is important for understanding the background and scope of the task. 
        Only use this context and do not make any assumptions or rely on prior knowledge.

        Then, thoroughly review the TensorFlow "{api_name}" API documentation provided in {documentation} in markdown format.
        This documentation contains the technical details and information you will need to reference when generating your customized response.

        Carefully read the user's question title {title} and body {body}. Make sure you understand the
        specific query or issue the user is asking about related to the TensorFlow "{api_name}" API. The issue type of this question
        is {issue_type}. And the definition of this issue type is {definition}.

        Finaly, carefully read and understand the task provided in {task}. This will give you the overall objective and guidelines for 
        the customized response you need to generate.

        Now you are ready to generate a customized response to the user's question based on the task, context, documentation, and the question itself:

        <customized_response>
        Write your customized response here. Ensure your response directly answers the user's query based on the guidelines in the task.
        </customized_response>

        <Format Output>
        Finally, format your complete output in markdown format with the following sections:

        # Customized Content

        Add the <cuztomized_response> here.

        ## Additional Resources
        Format the urls separating them with a title.

        #### Stack Overflow Q&A
        {qa}

        #### Related Web URLs
        {urls}

        #### Related Courses
        {course_urls}

        Only include the sections for Stack Overflow Q&A, Related Web URLs, and Related Courses if the corresponding variables are not empty.
        </Format Output>
    """

    prompt = [
        {
            "context": context,
            "title": title,
            "body": body,
            "issue_type": issue_type,
            "definition": definition,
            "api_name": api_name,
            "documentation": documentation,
            "task": task,
            "qa": qa,
            "urls": urls,
            "course_urls": course_urls,
        },
    ]

    PROMPT = PromptTemplate(
        template=template,
        input_variables=[
            "context",
            "title",
            "body",
            "issue_type",
            "definition",
            "api_name",
            "documentation",
            "task",
            "qa",
            "urls",
            "course_urls",
        ],
    )

    return PROMPT, prompt

