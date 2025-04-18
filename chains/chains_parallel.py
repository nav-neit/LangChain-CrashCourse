## task is to analyze the plot(chain1) and analyze the characters (chain2)
## tasks are done paralelly
## combine the results

from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from langchain.prompts import ChatPromptTemplate

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("GROQ API Key environment variable not set")

llm = ChatGroq(
    temperature=0,
    api_key=api_key,
    model =  'llama-3.3-70b-versatile'
)

## define output template for the movie summary
summary_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a movie critic."),
        ("human", "Provide a brief summary of the movie {movie_name}.")
    ]
)

## define the plot analysis
def analyze_plot(plot):
    plot_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a movie critic."),
            ("human", "Analyze the plot : {plot}. What are its strengths and weaknesses")
        ]
    )

    return plot_template.format_prompt(plot = plot)

## define character analysis
def analyze_characters(characters):
    character_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a movie critic."),
            ("human", "Analyze the characters : {characters}. What are their strengths and  weaknesses")
        ]
    )
    return character_template.format_prompt(characters = characters)

## combine analysis into a final verdict
def combine_verdicts(plot_analysis, character_analysis):
    return f"Plot Analysis: \n {plot_analysis}\n\n Character Analysis: \n{character_analysis}"

## simplify branches with LCEL
plot_branch_chain = (
    RunnableLambda(lambda x: analyze_plot(x)) | llm | StrOutputParser()
)

character_branch_chain = (
    RunnableLambda(lambda x: analyze_characters(x)) | llm | StrOutputParser()
)

chain = (
    summary_template
    | llm ## we prompt the llm to get the movie summary
    | StrOutputParser()
    | RunnableParallel(branches = {"plot" : plot_branch_chain, "characters" : character_branch_chain})
    | RunnableLambda(lambda x : combine_verdicts(x["branches"]["plot"], x["branches"]["characters"]))
)

## run the chain
result = chain.invoke({"movie_name" : "Inception"})

print(result)

## another example
## we want to write social media posts for various social media accounts, ex - X, LinkedIn, FaceBook
## collect the information
## chain - linkedin chain, twitter chain, facebook chain
## posting to the channels through the api
## or wite it to draft api , then user reviews it and then posts it

