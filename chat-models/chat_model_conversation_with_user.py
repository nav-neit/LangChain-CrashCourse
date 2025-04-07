from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("GROQ API KEY Environment variable not set")

llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    temperature = 0,
    groq_api_key = api_key
)

## all the chat history is stored in a loacal memory
chat_history = [] # using a list to store messages

## setting an initial system message
system_message = SystemMessage(content = "You are a helpful AI assistant.")
## adding system message to chat history
chat_history.append(system_message)


## Dynamically adding messages
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    ## add human message
    chat_history.append(HumanMessage(content = query))
    result = llm.invoke(chat_history)
    response = result.content
    ## add AI message
    chat_history.append(AIMessage(content = response))

    print(f"AI: {response}")

print("----------Message History-------------")

print(chat_history)