def return_template(noContext: bool = False):

    if noContext == True:
        template = (
            """
            You are an assistant with expertise in TensorFlow. Here are the steps to follow:\n

            Carefully read the question title and body: 
            \n ------- \n
            {title}
            {question}
            \n ------- \n

            Finaly, carefully read and understand the task provided below.
            \n ------- \n
            {task}
            \n ------- \n
            
            This will give you the overall objective and guidelines for the customized response you need to generate. Then perform the provided task.

            Your response will be added to the documentation. Maintain a similar tone to the documentation in your response.
            Add a concise description on what is answered in the response based on the question.

            Structure your answer as a description only, there shoud not be any separate code blocks.
            """
        )
    
    else:
        template = (
            """
            You are an assistant with expertise in TensorFlow. Here are the steps to follow:\n

            First, review the relevant information provided in context. This context is important for understanding the background and scope of the task. 
            Only use this context and do not make any assumptions or rely on prior knowledge. Below is the context:
            \n ------- \n
            {context}
            \n ------- \n

            Then, thoroughly review the TensorFlow API documentation provided below in markdown format.
            \n ------- \n
            {documentation}
            \n ------- \n

            Carefully read the question title and body: 
            \n ------- \n
            {title}
            {question}
            \n ------- \n

            Finaly, carefully read and understand the task provided below.
            \n ------- \n
            {task}
            \n ------- \n
            
            This will give you the guidelines for the response you need to generate. Then perform the provided task strictly following the instructions.

            Your response will be added to the documentation. Maintain a similar tone to the documentation in your response.
            Add a concise description on what is answered in the response based on the question.

            Answer the question based on the above provided context. \n
            Structure your answer as a description only, there shoud not be any separate code blocks.
            """
        )

    return template