
def return_prompt(title, body, issue_type, definition, api_name, documentation, task):
    template = """
        Use the following pieces of context as a support to answer the question at the end.  If cannot find relevant context, 
        please think rationally and answer from your own knowledge base.

        {context}

        Below is a Stack Overflow question posted by a user related to documentation replication on other examples.

        Question Title: {title}
        Question Body: {body}

        The above question is related to {issue_type} issue type. And below is the definition of that issue type.

        {issue_type} definition:
        {definition}


        Moreover, the question is related to the TensorFlow {api_name} API documentation. Below is the up-to-date TensorFlow 
        API documentation in markdown format:

        {documentation}


        {task}. Moreover, provide the response in markdown format in order to add that into the original documentation as a 
        new section "Alternative Resources". This customization should avoid any questions like this in future.

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
            },
        ]

    return template, prompt