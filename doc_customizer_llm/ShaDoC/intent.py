# ==========================================================================================================================
# IMPORT DEPENDENCIES
# ==========================================================================================================================

# from lib.config import COLOR

# LANGCHAIN MODULES
from langchain_cohere import ChatCohere
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_core.output_parsers import JsonOutputParser
# from selfcheckgpt.modeling_selfcheck_apiprompt import SelfCheckAPIPrompt

import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="/Users/sharukat/Documents/ResearchYU/Code/doc-customizer-llm/doc_customizer_llm/.env")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")   # Claude LLM API Key
# os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")

# selfcheck_prompt = SelfCheckAPIPrompt(client_type="openai", model="gpt-4o")

# ==========================================================================================================================
# MAIN CODE
# ==========================================================================================================================
class Output(BaseModel):
    intent: str = Field(description="intent or goal behind the question")
    # keywords: list = Field(description="API names within the question body")
    # sentences: list = Field(description="Split the final response in the intent variable into sentence by identifying the original sentence boundary")

def question_intent_identifier(title, body):
    # print(f"{COLOR['BLUE']}🚀: EXECUTING INTENT IDENTIFIER: Identifying intent of the SO question{COLOR['ENDC']}")

    responses = []

    parser = JsonOutputParser(pydantic_object=Output)
    format_instructions = parser.get_format_instructions()

    # And identify the API names within the question body.

# Your task is to identify the intent behind questions about the TensorFlow API documentation.
    template = """
        Your task is to identify the intent behind questions.

        First, read the question title:
        {title}

        Then read the full question body:
        {body}

        After reading the title and body, identify the intent or goal behind the question - what is the user ultimately trying to 
        accomplish or understand and emphasize the cause factor? 

        Do not make any external assumptions beyond the information provided in the title and body. Base the identified intent solely on 
        the given question text.

        {format_instructions}
    """

    PROMPT = PromptTemplate(
        template=template,
        input_variables=['title','body'], 
        partial_variables={"format_instructions":format_instructions})
    
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
    # llm=ChatCohere(model="command-r-plus", temperature=0)
    chain = PROMPT | llm | parser

    prompt = {"title" : title, "body" : body}

    response = chain.invoke(prompt)
    # for i in range(4):
    #     response = chain.invoke(prompt)
    #     responses.append(response['intent'])

    # sent_scores_prompt = selfcheck_prompt.predict(
    #     sentences = response['sentences'],  
    #     sampled_passages = responses, # list of sampled passages
    #     verbose = False,
    # )
    # print(sent_scores_prompt)

    # print(f"✅: EXECUTION COMPLETED\n")
    return response


# Test
# title = "Understanding tf.keras.layers.Dense()"
# question = """
# <p>I am trying to understand why there is a difference between calculating a dense layer operation directly and using the <code>keras</code> implementation.</p>
# <p>Following the documentation (<a href="https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense" rel="nofollow noreferrer">https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense</a>) <code>tf.keras.layers.Dense()</code> should implement the operation <code>output = activation(dot(input, kernel) + bias)</code> but <code>result</code> and <code>result1</code> below are not the same.</p>
# <pre class="lang-py prettyprint-override"><code>tf.random.set_seed(1)

# bias = tf.Variable(tf.random.uniform(shape=(5,1)), dtype=tf.float32)
# kernel = tf.Variable(tf.random.uniform(shape=(5,10)), dtype=tf.float32)
# x = tf.constant(tf.random.uniform(shape=(10,1), dtype=tf.float32))

# result = tf.nn.relu(tf.linalg.matmul(a=kernel, b=x) + bias)
# tf.print(result)

# test = tf.keras.layers.Dense(units = 5, 
#                             activation = 'relu',
#                             use_bias = True, 
#                             kernel_initializer = tf.keras.initializers.Constant(value=kernel), 
#                             bias_initializer = tf.keras.initializers.Constant(value=bias), 
#                             dtype=tf.float32)

# result1 = test(tf.transpose(x))

# print()
# tf.print(result1)

# </code></pre>
# <p>output</p>
# <pre class="lang-py prettyprint-override"><code>
# [[2.87080455]
#  [3.25458574]
#  [3.28776264]
#  [3.14319134]
#  [2.04760242]]

# [[2.38769 3.63470697 2.62423944 3.31286287 2.91121125]]

# </code></pre>
# <p>Using <code>test.get_weights()</code> I can see that the kernel and bias (<code>b</code>) are getting set to the correct values. I am using TF version 2.12.0.</p>

# """

# response = question_intent_identifier(title, question)
# print(response['intent'])