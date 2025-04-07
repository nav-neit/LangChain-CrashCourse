from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
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

## define a prompt template for animal facts
animal_facts_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You like telling facts and you tell facts about {animal}."),
        ("human", "Tell me {count} facts.")
    ]
)

## define a prompt template for translation to french
translation_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a translator and conert the provided into {language}"),
        ("human", "Translate the following text into {language}: {text}")
    ]
)

## define additional steps using RunnableLambda
count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")
prepare_for_translation = RunnableLambda(lambda output : {"text" : output, "language": "Spanish"})

## create a chain using LCEL
chain = animal_facts_template | llm | StrOutputParser() | \
prepare_for_translation | translation_template | llm | StrOutputParser()
# here we can add more runnable like posting it to a twitter post using the twitter api

## here the stroutputparser gives out a string we need to convert into a prompt template
## here the translation_template requires an object prepare_for_translation that needs to be passed into it
## run the chain

result = chain.invoke({"animal" : "cat", "count": 2})

print(result)