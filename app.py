from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
import google.generativeai as genai
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from fastapi.middleware.cors import CORSMiddleware
from utils import prioritize_sources
app = FastAPI()
os.environ['GOOGLE_API_KEY'] = "AIzaSyDh2gPu9X_96rpioBXcw7BQCDPZcFGMuO4"
genai.configure(api_key = os.environ['GOOGLE_API_KEY'])

from bot import chatbot

# Initialize the chatbot
mybot = chatbot()
workflow = mybot()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Define input schema
class Document(BaseModel):
    thread_id: str
    content: str

class MultiDocInput(BaseModel):
    docs: List[Document]

@app.post("/store-docs")
async def store_docs(input_data: MultiDocInput):
    """
    Store multiple documents in the chatbot.
    """
    for doc in input_data.docs:
        await query_bot(doc.content, doc.thread_id)
    return {"status": "success", "message": "Documents stored successfully."}




@app.get("/query")
async def query_bot(question: str, thread_id: str):
    """
    Query the chatbot with a question, using a specific thread ID.
    """
    try:
        input_data = {"messages": [HumanMessage(content=question)]}
        config = {"configurable": {"thread_id": thread_id}}

        # Invoke the workflow with the specified thread_id
        response = workflow.invoke(input_data, config)
        sources_url = []
        used_google_fact_check = response.get("used_google_fact_check", False)
        fact_check_results = response.get("fact_check_results", {})  # <— define early

        print(f"used_google_fact_check: {used_google_fact_check}")
        if isinstance(fact_check_results, dict):
            claims_list = fact_check_results.get("claims", [])
        else:
            claims_list = []
        if used_google_fact_check and claims_list:
            # fact_check_results = response.get("fact_check_results", [])
            print(f"fact_check_results: {fact_check_results}")
            # extract every review["url"] from each claim
            gc_urls = [
                review["url"]
                for claim in claims_list
                for review in claim.get("claimReview", [])
                if isinstance(review, dict) and "url" in review
            ]
            print(f"gc_urls: {gc_urls}")
            sources_url.extend(gc_urls)
        else:
            for tool_name, result in response.get("tool_results", {}).items():
                urls = result.get("sources_url")
                if isinstance(urls, list):
                    sources_url.extend(urls)
        
        result = response["messages"][-1].content

        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print(f"FACT CHECK RESULTS: {response['fact_check_results']}")
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

        print("\nConversation:")
        for message in response["messages"]:
            if isinstance(message, HumanMessage):
                print(f"\nHuman: {message.content}")
            elif isinstance(message, AIMessage):
                print(f"\nAI: {message.content}")
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    for tool_call in message.tool_calls:
                        print(f"  Tool Call: {tool_call['name']}({tool_call['args']})")

        sources = prioritize_sources(question, sources_url, result)
        response_payload = {
            "status": "success",
            "thread_id": thread_id,
            "response": result,
            "sources": sources[:3],
        }
        if used_google_fact_check and fact_check_results:
            response_payload["fact_check_results"] = fact_check_results
        return response_payload
    except KeyError:
        raise HTTPException(status_code=404, detail="Thread ID not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the chatbot API. Use /query to interact with the bot."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
