
from langchain.prompts import PromptTemplate

def return_prompt(context, title, body, issue_type, definition, api_name, documentation, task, qa, urls, course_urls):
    template = """
        Your task is to provide a customized response to a user's question based on the given context,
        documentation, and the task itself. Here are the steps to follow:

        <Review Context>
        First, review the context provided in {context}. This context is important for understanding the
        background and scope of the task. Only use this context and do not make any assumptions or rely on
        prior knowledge.
        </Review Context>

        <Review Documentation>
        Then, thoroughly review the TensorFlow "{api_name}" API documentation provided in {documentation} in markdown format.
        This documentation contains the technical details and information you will need to reference when
        generating your customized response.
        </Review Documentation>

        <Understand Question>
        Carefully read the user's question title {title} and body {body}. Make sure you understand the
        specific query or issue the user is asking about related to the TensorFlow "{api_name}" API. The issue type of this question
        is {issue_type}. And the definition of this issue type is {definition}.
        </Understand Question>

        <Understand the Task>
        Finaly, carefully read and understand the task provided in {task}. This will give you the overall
        objective and guidelines for the customized response you need to generate.
        </Understand the Task>

        <Generate Customized Response>
        Now you are ready to generate a customized response to the user's question based on the task,
        context, documentation, and the question itself:

        <scratchpad>
        Think through the steps needed to address the user's question and how you can utilize the provided
        context and documentation. Outline your approach here.
        </scratchpad>

        <customized_response>
        Write your customized response here. Ensure your response directly answers the user's query based on the guidelines 
        in the task.
        </customized_response>
        </Generate Customized Response>

        <Format Output>
        Finally, format your complete output in markdown format with the following sections:

        # Customized Content

        Add the <cuztomized_response> here.

        ## Additional Resources
        Format the urls separating them with a title.
        <QA>
        {qa}
        </QA>

        <Web URLs>
        {urls}
        </Web URLs>

        <Course URLs>
        {course_urls}
        </Course URLs>

        Only include the sections for QA, Web URLs, and Course URLs if the corresponding variables are not
        empty.
        </Format Output>
    """

    prompt = [
            {
                "context" : context,
                "title" : title,
                "body" : body,
                "issue_type" : issue_type,
                "definition" : definition,
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
        input_variables=['context', 'title','body', 'issue_type','definition', 'api_name',
                         'documentation', 'task', 'qa', 'urls', 'course_urls'])

    return PROMPT, prompt