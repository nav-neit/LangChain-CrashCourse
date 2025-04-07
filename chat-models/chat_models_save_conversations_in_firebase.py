##In production level environment we want ideally to stroe our historical conversations in some cloud  based sotrage systems
from dotenv import load_dotenv
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_groq import ChatGroq
import os

## Steps for fire store setup
## 1. create a fire base accounnt
## 2. Creare new firebase project and firestore database
## 3. Retrive the project id
## 4. Install Google Could CLI on our computer 
##    - Authenticate the google cloud cli with your google account (ADC)
## 5. install firestore pip install langchain-google-firestore
## 6. Enable the firestore api in google cloud console

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/Navaneeth/Desktop/LangChain-Tutorial/langchain-firestore-access.json"


PROJECT_ID = "langchain-204d5"
SESSION_ID = "user_session_new" # this could be a user name / unique id
COLLECTION_NAME = "chat_history"

## inititalize Firestore client
print("Initializing Firestore Client..")
client = firestore.Client(project = PROJECT_ID)

# The chat history list will now get stored in the cloud
# initialize FireStore Chat Message History
print("Initializing FireStore Chat Message History...")
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client,
)

print("Chat History Initialized")
print("Current Chat History:", chat_history.messages)

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("GROQ API KEY Environment variable not set")

llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    temperature = 0,
    groq_api_key = api_key
)

print("Start Chatting with the AI. Type 'exit' to quit")

while True:
    human_input = input("User: ")
    if human_input.lower() == "exit":
        break
    chat_history.add_user_message(human_input)

    ai_response = llm.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response.content)

    print(f"AI: {ai_response.content}")