def return_template():
    template = (
        """
        You are an assistant with expertise in TensorFlow. \n

        Your task is to provide a customized response to a user's question based on the given context. Here are the steps to follow:\n

        First, review the context provided in context. This context is important for understanding the background and scope of the task. 
        Only use this context and do not make any assumptions or rely on prior knowledge. Below is the context:
        \n ------- \n
        {context}
        \n ------- \n

        Then, thoroughly review the TensorFlow "{api_name}" API documentation provided below in markdown format.
        \n ------- \n
        {documentation}
        \n ------- \n

        This documentation contains the technical details and information you will need to reference when generating your customized response.

        Carefully read the user's question title and body: 
        \n ------- \n
        {title}
        {question}
        \n ------- \n

        Make sure you understand the specific query or issue the user is asking about related to the TensorFlow "{api_name}" API. 
        The issue type of this question is {issue_type}. And the definition of this issue type is {definition}.

        Finaly, carefully read and understand the task provided below.
        \n ------- \n
        {task}
        \n ------- \n
        
        This will give you the overall objective and guidelines for the customized response you need to generate. Then perform the provided task.

        Your response will be added to the documentation. Maintain a similar tone to the documentation in your response.
        Add a concise description on what is answered is the response based on the question.

        Answer the question based on the above provided context. \n
        Structure your answer as a description
        """
    )

    return template