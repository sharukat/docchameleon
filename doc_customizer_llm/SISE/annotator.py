from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
# from langchain_community.chat_models import ChatCohere
from langchain_cohere import ChatCohere
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="/Users/sharukat/Documents/ResearchYU/Code/doc-customizer-llm/doc_customizer_llm/.env")
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Sentence is meaningful on their own according to the criteria, 'yes' or 'no'")

def retrieval_grader():
    # LLM with function call
    llm = ChatCohere(model="command-r", temperature=0)
    parser = JsonOutputParser(pydantic_object=GradeDocuments)

    template = """
        You are a grader assessing meaninfulness of a sentence by strictly following the criteria.

        Below is the criteria:
        The following kinds of sentences should be rated as not being meaningful on their own:
            1. The sentence contains an explicit reference to another sentence.
                    Example: See the next step if you need a mutable list.
            2. The sentence contains references to "it", "this", "that", etc., which are not resolved within the sentence.
                    Example: It's a good way to produce a [n][3]-Matrix.
            3. The sentence is a question.
                    Example: Why Guava?
            4. The sentence is prefacing a code snippet (often indicated by a colon).
                    Example: You could create a factory method:
            5. The sentence is grammatically incomplete.
                    Example: So yes, ArrayList.
            6. The sentence contains communication between Stack Overflow users.
                    Example: Thanks to the comments I have to update my answer.
            7. The sentence references code elements that come from user examples rather than the API.
                    Example: You might be surprised to find out that in your sample code, s1 == s2 returns true!
            8. The sentence references specific Stack Overflow users.
                    Example: Like EJP said the key part is to save in a byte array.
            9. The sentence only contains a link.
                    Example: http://www.jmagick.org/index.html.
            10. The sentence contains a reference to something that is not an obvious part of the API ("block" in the example)
                    Example: The block size is parameterized for run-time performance optimization.
            11. The sentences starts with "but", "and", "or" etc.
                    Example: And then another instance of string with content Hello world.
            12. The sentence is a generic statement that is unrelated to the API type.
                    Example: Most random number generators are, in fact, pseudo random.
            13. The sentence contains a comparison that is incomplete (i.e., one part of the comparison is missing).
                    Example: char[] is less vulnerable.
            14. The sentence resulted from a parsing error.
                    Example: Because the java.io.
            15. The sentence requires another sentence to be complete.
                    Example: First, you have to know the encoding of string that you want to convert.
            16. The sentence contains an explicit reference to a piece of context that's missing.
                    Example: What you're trying to do here is rather unusual, to say the least.
        
        
        Other sentences are considered meaningful.

        Below is the input sentence:  
        {sentence}    

        Give a binary score 'yes' or 'no' score to indicate whether the setence is meaningful or not.

        Strictly follow the format instructions given below making sure the output is in json format.
        {format_instructions}
    """

    grade_prompt = PromptTemplate(
        template=template,
        input_variables=["sentence"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    grader = grade_prompt | llm | parser
    return grader

# sentence = '<p>You can use the following instruction:</p>\n<pre><code>new ArrayList&lt;&gt;(Arrays.asList(array));\n</code></pre>'
# rg = retrieval_grader()
# score = rg.invoke({"sentence": sentence})
# print(score)