from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
import google.generativeai as genai
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from fastapi.middleware.cors import CORSMiddleware
from tools import ArticleTools
from FAQVectorStore import store_faq, search_faq
from utils import prioritize_sources,check_boom_verification_status,store_unverified_content_to_sheets,test_google_sheets_manually
app = FastAPI(debug=True)
os.environ['GOOGLE_API_KEY'] = "AIzaSyDh2gPu9X_96rpioBXcw7BQCDPZcFGMuO4"
genai.configure(api_key = os.environ['GOOGLE_API_KEY'])

from bot import chatbot

# Initialize the chatbot
mybot = chatbot()
article_tools = ArticleTools()
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
class FAQInput(BaseModel):
    question: str
    answer: str
class FAQItem(BaseModel):
    question: str
    answer: str
class FAQBatch(BaseModel):
    faqs: List[FAQItem]
class FAQQuery(BaseModel):
    query: str
    top_k: int = 3
class ScamCheckInput(BaseModel):
    query: str

@app.post("/store-docs")
async def store_docs(input_data: MultiDocInput):
    """
    Store multiple documents in the chatbot.
    """
    for doc in input_data.docs:
        await query_bot(doc.content, doc.thread_id)
    return {"status": "success", "message": "Documents stored successfully."}




# @app.get("/query")
# async def query_bot(question: str, thread_id: str, using_Twitter: bool = False, using_Whatsapp: bool = False):
#     """
#     Query the chatbot with a question, using a specific thread ID.
#     """
#     try:
#         print("USING WHATSAPP",using_Whatsapp)
#         chatbot_type = "whatsapp" if using_Whatsapp else ("twitter" if using_Twitter else "web")
#         input_data = {"messages": [HumanMessage(content=question)], "isTwitterMsg": using_Twitter, "isWhatsappMsg": using_Whatsapp, "chatbot_type": chatbot_type}
#         config = {"configurable": {"thread_id": thread_id}}

#         # Invoke the workflow with the specified thread_id
#         response = workflow.invoke(input_data, config)
#         sources_url = []
#         used_google_fact_check = response.get("used_google_fact_check", False)
#         fact_check_results = response.get("fact_check_results", {})  # <— define early
#         used_general_search= response.get("used_general_search", {})
#         general_search_results= response.get("general_search_results", {})

#         print(f"used_google_fact_check: {used_google_fact_check}")
#         if isinstance(fact_check_results, dict):
#             claims_list = fact_check_results.get("claims", [])
#         else:
#             claims_list = []
#         if used_google_fact_check and claims_list:
#             # fact_check_results = response.get("fact_check_results", [])
#             print(f"fact_check_results: {fact_check_results}")
#             # extract every review["url"] from each claim
#             gc_urls = [
#                 review["url"]
#                 for claim in claims_list
#                 for review in claim.get("claimReview", [])
#                 if isinstance(review, dict) and "url" in review
#             ]
#             print(f"gc_urls: {gc_urls}")
#             sources_url.extend(gc_urls)
#         else:
#             for tool_name, result in response.get("tool_results", {}).items():
#                 urls = result.get("sources_url")
#                 if isinstance(urls, list):
#                     sources_url.extend(urls)
        
#         result = response["messages"][-1].content
#         print(f"Result: {result}")
#         isBoomVerified = check_boom_verification_status(result)

#         print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
#         print(f"FACT CHECK RESULTS: {response['fact_check_results']}")
#         print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

#         print("\nConversation:")
#         for message in response["messages"]:
#             if isinstance(message, HumanMessage):
#                 print(f"\nHuman: {message.content}")
#             elif isinstance(message, AIMessage):
#                 print(f"\nAI: {message.content}")
#                 if hasattr(message, 'tool_calls') and message.tool_calls:
#                     for tool_call in message.tool_calls:
#                         print(f"  Tool Call: {tool_call['name']}({tool_call['args']})")
#         if used_general_search and 'trusted_results' in general_search_results:
#             sources_url = [
#                 result['url']
#                 for result in general_search_results['trusted_results']
#                 if 'url' in result
#             ]
#         sources = prioritize_sources(question, sources_url, result)

#         # Store unverified content in Google Sheets
#         if not isBoomVerified:
#             print("⚠️  CONTENT NOT VERIFIED - Storing in Google Sheets...")
#             await store_unverified_content_to_sheets(
#                 question=question,
#                 response=result,
#                 thread_id=thread_id,
#                 fact_check_results=fact_check_results,
#                 sources=sources[:3],
#                 using_Twitter=using_Twitter,
#                 using_Whatsapp=using_Whatsapp
#             )
#         else:
#             print("✅ CONTENT VERIFIED - Not storing in Google Sheets")
#         response_payload = {
#             "status": "success",
#             "thread_id": thread_id,
#             "response": result,
#             "sources": sources[:3],
#             "isBoomVerified": isBoomVerified,
#         }
#         if used_google_fact_check and fact_check_results:
#             response_payload["fact_check_results"] = fact_check_results
            
#         print("ENVIRONMENT VARIABLES IN OS",os.environ)
#         return response_payload
#     except KeyError:
#         raise HTTPException(status_code=404, detail="Thread ID not found.")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# Add this to your /query endpoint - just before the response_payload creation

@app.get("/query")
async def query_bot(
    question: str, 
    thread_id: str, 
    using_Twitter: bool = False, 
    using_Whatsapp: bool = False,
    using_ScamCheck: bool = False  # ADD THIS PARAMETER
):
    """
    Query the chatbot with a question, using a specific thread ID.
    """
    try:
        print("USING WHATSAPP", using_Whatsapp)
        print("USING SCAM CHECK", using_ScamCheck)  # ADD THIS
        
        # MODIFY THIS SECTION - treat scam_check as web
        if using_ScamCheck:
            chatbot_type = "web"  # Keep as web for same flow
            is_scam_check = True   # But track that it's scam check
        elif using_Whatsapp:
            chatbot_type = "whatsapp"
            is_scam_check = False
        elif using_Twitter:
            chatbot_type = "twitter"
            is_scam_check = False
        else:
            chatbot_type = "web"
            is_scam_check = False
            
        input_data = {
            "messages": [HumanMessage(content=question)], 
            "isTwitterMsg": using_Twitter, 
            "isWhatsappMsg": using_Whatsapp,
            "chatbot_type": chatbot_type
        }
        config = {"configurable": {"thread_id": thread_id}}

        # Invoke the workflow with the specified thread_id
        response = workflow.invoke(input_data, config)
        sources_url = []
        used_google_fact_check = response.get("used_google_fact_check", False)
        fact_check_results = response.get("fact_check_results", {})
        used_general_search = response.get("used_general_search", {})
        general_search_results = response.get("general_search_results", {})

        print(f"used_google_fact_check: {used_google_fact_check}")
        if isinstance(fact_check_results, dict):
            claims_list = fact_check_results.get("claims", [])
        else:
            claims_list = []
            
        if used_google_fact_check and claims_list:
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

        print("$" * 100)
        print(f"FACT CHECK RESULTS: {response['fact_check_results']}")
        print("$" * 100)

        print("\nConversation:")
        for message in response["messages"]:
            if isinstance(message, HumanMessage):
                print(f"\nHuman: {message.content}")
            elif isinstance(message, AIMessage):
                print(f"\nAI: {message.content}")
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    for tool_call in message.tool_calls:
                        print(f"  Tool Call: {tool_call['name']}({tool_call['args']})")
                        
        if used_general_search and 'trusted_results' in general_search_results:
            sources_url = [
                result['url']
                for result in general_search_results['trusted_results']
                if 'url' in result
            ]
        
        # ADD THIS NEW SECTION - Extract URLs from response text for scam check
        if is_scam_check and not sources_url:
            import re
            # Extract all URLs from the response text
            url_pattern = r'https?://[^\s\)\]>]+'
            extracted_urls = re.findall(url_pattern, result)
            if extracted_urls:
                sources_url.extend(extracted_urls)
                print(f"Extracted {len(extracted_urls)} URLs from scam check response")
        
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
            
        print("ENVIRONMENT VARIABLES IN OS", os.environ)
        return response_payload
        
    except KeyError:
        raise HTTPException(status_code=404, detail="Thread ID not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from fastapi import Query
from typing import List, Dict, Any, Optional

def general_query_search(query: str, language_code: str = "en") -> Dict[str, Any]:
        """
        Performs a search using SerpAPI and filters results to return only trusted sources.

        Parameters:
        - query: The user's general query (not a factual claim)
        - language_code: ISO code for language (e.g., "en", "hi", "bn")

        Returns:
        - Dict with filtered result list containing 'title', 'url', and 'snippet'
        """

        import requests
        import os

        serp_api_key = os.getenv("SERP_API_KEY")
        url = "https://serpapi.com/search"
        print("SERP API: ", serp_api_key)
        params = {
            "q": query,
            "location": "India",
            "hl": language_code,
            "gl": "in",
            "api_key": serp_api_key,
            "num": 10
        }

        response = requests.get(url, params=params)
        data = response.json()
        print("DATA",data)
        trusted_domains = [
            "bbc.com/hindi", "bbc.com/marathi", "bbc.com/news/world/asia/india",
            "indianexpress.com", "thenewsminute.com", "thehindu.com",
            "indiaspendhindi.com", "indiaspend.com"
        ]

        results = []
        if "organic_results" in data:
            for item in data["organic_results"]:
                url = item.get("link", "")
                if any(domain in url for domain in trusted_domains):
                    results.append({
                        "title": item.get("title", ""),
                        "url": url,
                        "snippet": item.get("snippet", "")
                    })

        return {"trusted_results": results}
   

@app.get("/test-serp-api")
async def test_serp_api(
    query: str = Query(..., description="Search query"),
    language_code: str = Query("en", description="Language code (en/hi/bn)")
):
    """
    Test the SerpAPI-based general_query_search tool. 
    Returns trusted search results from SerpAPI for the given query.
    """
    try:
        result = general_query_search(query,language_code)
        return {"status": "success", "results": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
    

@app.post("/faq/store")
async def store_faq_route(faq: FAQInput):
    """
    Store a single FAQ in the vector store.
    """
    try:
        result = store_faq(faq.question, faq.answer)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/faq/search")
async def search_faq_route(query: str, top_k: int = 3):
    """
    Search FAQs in the vector store.
    """
    try:
        result = search_faq(query=query, top_k=top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/faq/store-bulk")
async def store_bulk_faqs(batch: FAQBatch):
    """
    Store multiple FAQs in the vector store.
    """
    results = []
    for faq in batch.faqs:
        try:
            result = store_faq(faq.question, faq.answer)
            results.append({"question": faq.question, "status": "success", "id": result["id"]})
        except Exception as e:
            results.append({"question": faq.question, "status": "error", "message": str(e)})
    return {"stored": len(results), "details": results}

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
@app.post("/scam-check")
async def scam_check(input_data: ScamCheckInput):
    """
    Perform scam check using semantic search and return filtered ScamCheck URLs.
    """
    try:
        result = article_tools.scam_check_search(
            query=input_data.query,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the chatbot API. Use /query to interact with the bot."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
