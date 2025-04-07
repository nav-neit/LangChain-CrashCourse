from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("GROQ  API KEY Environment variable not set")

llm = ChatGroq(
    temperature = 0,
    api_key= api_key,
    model = 'llama-3.3-70b-versatile'
)

## prompt with customized system and human messages given as tuples
messages = [
    (
        "system", "You are comedian who tells jokes about {topic}." 
    ),
    (
        "human" , "Tell me {joke_count} jokes."
    )
]

## convert this messages to a prompt that langchain understands
propmt_template = ChatPromptTemplate.from_messages(messages)

## populate prompt template with keywords using invoke
prompt = propmt_template.invoke(
    {
        "topic" : "Lawyers",
        "joke_count" : 3
    }
)

result = llm.invoke(prompt)

print(result.content)

