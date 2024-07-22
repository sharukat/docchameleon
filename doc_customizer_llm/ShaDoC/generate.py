from ShaDoC.templates.template_2 import return_template
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

def generate_explanation(title, question, documentation=None, context = None, noContext = False):
    template = return_template(noContext)
    print(f"{'='*10} 🛠 GENERATE SOLUTION {'='*10}\n")

    if noContext == True:
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        task = "Your task is to generate an comprehensive explanation description only to address the question. Again, only explanation description only. No separete code examples should be included."

        prompt = PromptTemplate(
            template=template,
            input_variables=['title', 'question', 'task'])

        chain =  prompt | llm

        solution = chain.invoke({
            "title": title,
            "question": question,
            "task": task,
        })

    else:
        llm = ChatOpenAI(temperature=0, model="gpt-4o")
        task = "Your task is to generate an comprehensive explanation description only to address the question by only using the knowledge provided as 'context'. Again, only explanation description only. No separete code examples should be included."

        prompt = PromptTemplate(
        template=template,
        input_variables=['context', 'documentation', 'title', 'question', 'task'])

        chain = prompt | llm

        solution = chain.invoke(
            {
                "context":context,
                "documentation": documentation,
                "title": title,
                "question": question,
                "task": task,
            }
        )

    return solution