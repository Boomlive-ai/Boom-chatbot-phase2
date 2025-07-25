from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
import google.generativeai as genai
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from fastapi.middleware.cors import CORSMiddleware
from utils import prioritize_sources,check_boom_verification_status,store_unverified_content_to_sheets,test_google_sheets_manually
app = FastAPI(debug=True)
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
async def query_bot(question: str, thread_id: str, using_Twitter: bool = False, using_Whatsapp: bool = False):
    """
    Query the chatbot with a question, using a specific thread ID.
    """
    try:
        print("USING WHATSAPP",using_Whatsapp)
        chatbot_type = "whatsapp" if using_Whatsapp else ("twitter" if using_Twitter else "web")
        input_data = {"messages": [HumanMessage(content=question)], "isTwitterMsg": using_Twitter, "isWhatsappMsg": using_Whatsapp, "chatbot_type": chatbot_type}
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
        print(f"Result: {result}")
        isBoomVerified = check_boom_verification_status(result)

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

        # Store unverified content in Google Sheets
        if not isBoomVerified:
            print("⚠️  CONTENT NOT VERIFIED - Storing in Google Sheets...")
            await store_unverified_content_to_sheets(
                question=question,
                response=result,
                thread_id=thread_id,
                fact_check_results=fact_check_results,
                sources=sources[:3],
                using_Twitter=using_Twitter,
                using_Whatsapp=using_Whatsapp
            )
        else:
            print("✅ CONTENT VERIFIED - Not storing in Google Sheets")
        response_payload = {
            "status": "success",
            "thread_id": thread_id,
            "response": result,
            "sources": sources[:3],
            "isBoomVerified": isBoomVerified,
        }
        if used_google_fact_check and fact_check_results:
            response_payload["fact_check_results"] = fact_check_results
        return response_payload
    except KeyError:
        raise HTTPException(status_code=404, detail="Thread ID not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# @app.get("/test-google-sheets")
# async def test_google_sheets_endpoint():
#     """Test endpoint to manually trigger Google Sheets storage"""
#     try:
#         success = await test_google_sheets_manually()
#         return {
#             "status": "success" if success else "failed",
#             "message": "Check your Google Drive for 'Unverified Content Tracker' spreadsheet" if success else "Test failed - check server logs"
#         }
#     except Exception as e:
#         return {"status": "error", "message": str(e)}

@app.get("/")
async def root():
    return {"message": "Welcome to the chatbot API. Use /query to interact with the bot."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
