## we can customize the chain with our own methods and usecases
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("GROQ API KEY environment variable not set")

llm = ChatGroq(
    temperature=0,
    model = 'llama-3.3-70b-versatile',
    api_key=api_key
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You love facts and you will tell facts about {animal}"),
        ("human", "Tell me {count} facts.")
    ]
)

## runnable lambda -  its lets us create a wrapper for each task so as we can use it as a single reusable unit
## each runnable lambda takes an input does computation and returns an output

## we have to provide a function inside the runnable lambda
## Task-1/Runnable-1
format_prompt = RunnableLambda(lambda x : prompt_template.format_prompt(**x))
## here the format_prompt replaces only the placeholder values in the prompt
## invoke did replace the placeholder values + changes the prompt into format suitable to be passed to the LLM

## Task-2/Runnable-2
invoke_model = RunnableLambda(lambda x: llm.invoke(x.to_messages()))

## Task-3 Runnable -3
parse_output = RunnableLambda(lambda x : x.content)

## we need to chain these tasks into single unifiedd workflow
## first - first task, last - last task, middle - all the tasks in the middle should be provided as an array format
chain = RunnableSequence(first = format_prompt, middle=[invoke_model],
                         last = parse_output)
## LCEL uses the pipe operator | to combine different runnables
response = chain.invoke({
    "animal" : "cat",
    "count":2
})

print(response)