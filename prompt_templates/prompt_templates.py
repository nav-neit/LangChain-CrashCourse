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

## prompt templates with placeholders
template = "Write a {tone} email to {company} expressing interest " \
"in the {position} position mentioning {skill} as a key strength." \
"Keep it to 4 lines max"

## convert this prompt to a prompt that langchain understands

propmt_template = ChatPromptTemplate.from_template(template)

## populate prompt template with keywords using invoke
prompt = propmt_template.invoke(
    {
        "tone" : "Energetic",
        "company" : "Samsung",
        "position" : "AI Engineer",
        "skill" : "AI"
    }
)

result = llm.invoke(prompt)

print(result.content)

