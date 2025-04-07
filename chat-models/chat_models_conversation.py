from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("GROQ API Environment variable not set")

llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    temperature = 0,
    groq_api_key = api_key
)

## hardcoding a conversation history

messages = [
    # this defines the ai's role
    SystemMessage("You are an expert in Social Media Content Strategy"),

    # this is the question, content we want to get
    HumanMessage("Give a short tip to create engaging posts on LinkedIn")

    # we can add more to this more human messages , AI messages etc

]

response = llm.invoke(messages)
print(response.content)