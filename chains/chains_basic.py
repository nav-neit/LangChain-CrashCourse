from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import os

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("GROQ API KEY Environment Variable not set")

llm = ChatGroq(
    temperature = 0,
    api_key= api_key,
    model = 'llama-3.3-70b-versatile'
)

## define prompt templates 
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a facts expert who knows facts about "
        "{animal}."),
        ("human", "Tell me {fact_count} facts.")
    ]
)

## task1 - populate all the placeholders and create the final prommpt
## task2 - pass the prompt to the model

## using chains we can remove calling invoke multiple times
## chaining using the pipe | operator
## combined chain created using Langchain Expression Language (LCEL)
## stroutputparser extracts the content only from the response
chain = prompt_template | llm | StrOutputParser()

## running the chain
## the placeholders we pass will be available all the across the chain
result = chain.invoke(
    {
    "animal": "dog", "fact_count" : 2
    }
)

print(result)

