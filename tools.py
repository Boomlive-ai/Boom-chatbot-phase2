from langchain_core.tools import tool, InjectedToolArg
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from datetime import datetime, timedelta
import re, os
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from utils import fetch_custom_range_articles_urls, fetch_latest_article_urls,extract_articles, prioritize_sources, fetch_google_fact_check_urls, fetch_serp_trusted_urls
from utils import prioritize_sources, translate_text, general_query_search
from FAQVectorStore import search_faq
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import requests
load_dotenv() 

from langchain.schema import Document
class ArticleTools:
    def __init__(self):
        pass
   

    @staticmethod
    def scam_check_search(query: str) -> dict:
        """
        Performs semantic search using the latest index to retrieve scam-related article URLs.
        Filters results to only include unique BoomLive ScamCheck URLs.
        If none found, fetches articles from Google Fact Check, SerpAPI, and BoomLive API in parallel.

        Parameters:
        - query: The scam-related claim or keyword to verify.

        Returns:
        - Dictionary with filtered and prioritized article URLs.
        """

        print("🔍 Scam Check Search Triggered")
        print("Query:", query)

        # Initialize latest index retriever
        latest_index = PineconeVectorStore(
            index_name=os.getenv("PINECONE_LATEST_INDEX_NAME"),
            embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
            pinecone_api_key="829bebca-aceb-4416-8e78-b1972af62abc"
        )
        retriever = latest_index.as_retriever(search_kwargs={"k": 5})
        documents = retriever.get_relevant_documents(query)

        # Extract and filter ScamCheck sources
        sources = [doc.metadata.get("source", "Unknown") for doc in documents]
        scamcheck_sources = {
            url for url in sources
            if "boomlive.in/decode/scamcheck/" in url
        }

        # Prioritize ScamCheck URLs
        prioritized = prioritize_sources(query, list(scamcheck_sources))
        print(f"✅ Retrieved {len(prioritized)} unique ScamCheck URLs from vector store")

        # Fetch from BoomLive API in parallel with other sources
        def fetch_boomlive_api_urls(query_text: str) -> list:
            """Fetches article URLs from BoomLive ScamCheck API"""
            try:
                api_url = "https://toolbox.boomlive.in/scamcheck-article/submit_form.php"
                params = {
                    "action": "search",
                    "message": query_text
                }
                response = requests.get(api_url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                if data.get("status") == "success" and data.get("data"):
                    urls = [article["article_url"] for article in data["data"]]
                    print(f"✅ BoomLive API returned {len(urls)} URLs")
                    return urls
                return []
            except Exception as e:
                print(f"⚠️ BoomLive API error: {str(e)}")
                return []

        # Combine vector store results with API results
        with ThreadPoolExecutor() as executor:
            future_api = executor.submit(fetch_boomlive_api_urls, query)
            api_urls = future_api.result()

        # Combine and deduplicate all BoomLive sources
        all_boomlive_urls = list(set(prioritized + api_urls))
        
        # Re-prioritize combined sources
        if all_boomlive_urls:
            final_prioritized = prioritize_sources(query, all_boomlive_urls)
            print(f"✅ Returning {len(final_prioritized)} combined BoomLive URLs")
            return {"sources_url": final_prioritized}

        # If no BoomLive results at all, fetch from fallback sources
        print("⚠️ No BoomLive results found. Fetching from fallback sources...")

        with ThreadPoolExecutor() as executor:
            future_factcheck = executor.submit(fetch_google_fact_check_urls, query)
            future_serp = executor.submit(fetch_serp_trusted_urls, query, "en")

            factcheck_urls = future_factcheck.result()
            serp_urls = future_serp.result()

        combined_urls = list(set(factcheck_urls + serp_urls))
        prioritized_fallback = prioritize_sources(query, combined_urls)

        print(f"✅ Fallback sources retrieved: {len(prioritized_fallback)}")
        return {"sources_url": prioritized_fallback}

    @staticmethod
    @tool
    def faq_scam_search(query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Searches the FAQ vector store for relevant answers for scam seacrh questions.

        **USE THIS TOOL WHEN THE QUERY MATCHES OR RELATES TO ANY OF THE FOLLOWING:**
        - How Scammers Trick People on Social Media
        - How to Detect and Handle Online Impersonation
        - How to Spot Fake Messages on WhatsApp
        - How to Tell if an Online Giveaway or Contest Is Fake
        - How to Spot Fake Job Offers Online
        - What to Do If You Click a Suspicious Link
        - How to Spot Dangerous Apps or APKs
        - How to Know if a Shopping Website Is Genuine
        - How Scammers Trick People in Banking Scams
        - How to Check if a Payment Link or QR Code Is Safe
        - What to Do If You Get a Fake Bank Call or SMS
        - How to Spot Fake UPI Payment Requests
        - How Fake Loan and Investment Scams Work
        - What Is A “Digital Arrest”? Is It Legal?
        - How to Stay Safe From “Digital Arrest”?
        - How to Confirm if Something Is a Scam Safely
        - How to Keep Your Personal Information Safe
        - What to Do If You Shared OTP or PIN by Mistake
        - Common Mistakes People Make with Scams
        - How to Tell If a Charity Appeal Is Fake
        - What Are Fake Tech Support Scams
        - How Lottery and Prize Scams Work
        - How to Tell If a Family Emergency Message Is Real
        - How to Tell If an Email Is Real


        This tool is optimized for scam-related FAQs and fraud awareness topics. It performs semantic search over a curated vector store of verified answers.

        Parameters:
        - query: The user's question
        - top_k: Number of top results to return

        Returns:
        - Dict with matched FAQs including question, answer, and score
        """
        return search_faq(query=query, top_k=top_k)
    @staticmethod
    @tool
    def general_query_search(query: str, language_code: str = "en") -> Dict[str, Any]:
        """
        Performs a general web search for informational queries, how-to questions, definitions, explanations, and general knowledge requests.

        **USE THIS TOOL FOR GENERAL QUERIES:**
        - General information requests ("What is climate change?", "How to cook rice?")
        - Educational queries ("Explain photosynthesis", "History of India")
        - How-to and instructional queries ("How to apply for passport?", "Steps to start a business")
        - Definition requests ("What does AI mean?", "Define democracy")
        - General knowledge questions ("Who invented the telephone?", "Capital of France")
        - Opinion-seeking queries ("Best places to visit in India", "Top universities")
        - Comparative queries ("Difference between Android and iOS")
        - Process explanations ("How does voting work?", "What is the procedure for...")
        - General current affairs without specific factual claims
        - Queries asking for lists, recommendations, or general information
        - Questions starting with "What", "How", "Why", "Where", "When" that seek general information
        - Educational content requests in any language

        **DO NOT USE FOR:**
        - Specific factual claims that need verification
        - Statements containing statistics, numbers, or percentages about recent events
        - Claims about arrests, executions, government actions, or conflicts
        - Content that appears to be from social media or news sources needing fact-checking
        - Verification requests ("Is this true?", "Did this happen?")

        Parameters:
        - query: The user's general information query
        - language_code: ISO code for language (e.g., "en", "hi", "bn")

        Returns:
        - Dict with filtered result list containing 'title', 'url', and 'snippet' from trusted sources
        """
        import requests
        import os

        serp_api_key = os.getenv("SERP_API_KEY")
        url = "https://serpapi.com/search"

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
        print("Results: ",results)
        return {"trusted_results": results}
    
    
    @staticmethod
    @tool
    def rag_search(query: str, language_code: str, original_message: str = None, chatbot_type: str = "web") -> str: ## Change name to fact_check_claim_search
        """
            Performs semantic searches across news and fact-check articles to verify specific factual claims and statements.

        **USE THIS TOOL ONLY FOR FACT-CHECKING AND VERIFICATION:**
        - Specific factual claims that need verification ("7000 Jews were arrested")
        - Statements with specific numbers, statistics, or percentages about recent events
        - Claims about government actions, arrests, executions, or military activities
        - Content that appears to be forwarded from social media or news sources
        - Verification requests with phrases like "क्या यह सच है?", "Is this true?", "Did this happen?"
        - Claims about conflicts, politics, or controversial recent events
        - Statements about treatment of religious or ethnic minorities
        - Content with hashtags, social media handles, or news-like format
        - Claims about specific incidents, arrests, or government decisions
        - Statements that read like news reports or social media posts

        **SPECIFIC INDICATORS THAT REQUIRE FACT-CHECKING:**
        - Exact numbers with context (e.g., "7000 यहूदी", "700 से ज्यादा")
        - Government/military references with specific claims ("सेना ने गिरफ्तार किया")
        - Action verbs indicating specific events ("गिरफ्तार", "लटकाया", "आरोप लगाया")
        - Geographic references with specific claims ("ईरान में हुई गिरफ्तारी")
        - Claims about persecution, arrests, or violence with specifics
        - Statements that make definitive assertions about recent events

        **DO NOT USE FOR:**
        - General information queries ("What is democracy?", "How to cook?")
        - Educational questions ("Explain climate change")
        - How-to queries ("How to apply for visa?")
        - Definition requests ("What does AI mean?")
        - General current affairs without specific claims
        - Opinion-seeking queries ("Best places to visit")
        - Process explanations without factual claims to verify

        Parameters:
        - query: Extract the specific factual claim that needs verification
        - language_code: "en" for English, "hi" for Hindi, "bn" for Bangla
        - original_message: The complete, unprocessed user input exactly as received
        - chatbot_type: Type of chatbot interface ("web", "whatsapp", "twitter")
        
        Returns:
        - Relevant articles and sources that can verify or fact-check the specific claims
        """
        
        print("Inside rag_search",query, language_code, original_message)
        query = original_message if original_message else query
    # Prepare queries for each language (use provided translations or fallback to original)
        english_search_query = translate_text(query, "en")
        hindi_search_query = translate_text(query, "hi")
        bangla_search_query = translate_text(query, "bn")
        print("***********************************************************************")
        print("Chatbot Type:", chatbot_type)
        print("English Query:", english_search_query)
        print("Hindi Query:", hindi_search_query)
        print("Bangla Query:", bangla_search_query)
        print("***********************************************************************")

        # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        # print(f"Query: {query}, Language Code: {language_code}")
        # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

        latest_index = PineconeVectorStore(
            index_name=os.getenv("PINECONE_LATEST_INDEX_NAME"),
            embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
            pinecone_api_key="829bebca-aceb-4416-8e78-b1972af62abc"
        )
        old_index = PineconeVectorStore(
            index_name=os.getenv("PINECONE_OLD_INDEX_NAME"),
            embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
            pinecone_api_key="829bebca-aceb-4416-8e78-b1972af62abc"
        )
        hindi_index = PineconeVectorStore(
            index_name=os.getenv("PINECONE_HINDI_INDEX_NAME"),
            embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
            pinecone_api_key="829bebca-aceb-4416-8e78-b1972af62abc"
        )
        bangla_index = PineconeVectorStore(
            index_name=os.getenv("PINECONE_BANGLA_INDEX_NAME"),
            embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
            pinecone_api_key="829bebca-aceb-4416-8e78-b1972af62abc"
        )
        index_to_use = "both"
        if language_code=='hi':
            index_to_use='hindi-boom-articles'
        elif language_code=='bn':
            index_to_use='bangla-boom-articles'
        if chatbot_type == "whatsapp":
            index_to_use = "all"
        all_docs = []
        all_sources = []
        
        # Simple mock implementation of semantic search
        # Clean up index_to_use if needed
        if index_to_use is not None:
            index_to_use = index_to_use.split(".")[-1].strip()
        
        # Retrieve documents based on selected index
        if index_to_use in ["latest", None, "both", "all"]:
            latest_retriever = latest_index.as_retriever(search_kwargs={"k": 5})
            latest_docs = latest_retriever.get_relevant_documents(english_search_query)
            all_docs.extend(latest_docs)
            all_sources.extend([doc.metadata.get("source", "Unknown") for doc in latest_docs])
            print(f"Latest documents retrieved: {len(latest_docs)}")
        
        if index_to_use in ["old", "both", "all"]:
            old_retriever = old_index.as_retriever(search_kwargs={"k": 5})
            old_docs = old_retriever.get_relevant_documents(english_search_query)
            all_docs.extend(old_docs)
            all_sources.extend([doc.metadata.get("source", "Unknown") for doc in old_docs])
            print(f"Old documents retrieved: {len(old_docs)}")
        
        if index_to_use in ["hindi-boom-articles", "all"]:
            hindi_retriever = hindi_index.as_retriever(search_kwargs={"k": 5})
            hindi_docs = hindi_retriever.get_relevant_documents(hindi_search_query)
            all_docs.extend(hindi_docs)
            all_sources.extend([doc.metadata.get("source", "Unknown") for doc in hindi_docs])
            print(f"Hindi documents retrieved: {len(hindi_docs)}")
        
        if index_to_use in ["bangla-boom-articles", "all"]:
            bangla_retriever = bangla_index.as_retriever(search_kwargs={"k": 5})
            bangla_docs = bangla_retriever.get_relevant_documents(bangla_search_query)
            all_docs.extend(bangla_docs)
            all_sources.extend([doc.metadata.get("source", "Unknown") for doc in bangla_docs])
            print(f"Bangla documents retrieved: {len(bangla_docs)}")
            
        # Perform general search and collect trusted URLs
        search_results = general_query_search(query, language_code)
        print("SEARCH RESULTS", search_results)

        if search_results and search_results.get('trusted_results'):
            print(f"Found {len(search_results['trusted_results'])} general search results")
            
            # Append general search result URLs to all_sources
            general_sources = [result['url'] for result in search_results['trusted_results']]
            all_sources.extend(general_sources)

            # Optionally, convert search results to document-like format for consistency
            for result in search_results['trusted_results']:
                doc = Document(
                    page_content=result.get('snippet', ''),
                    metadata={"source": result.get('url', 'Unknown'), "title": result.get('title', '')}
                )
                all_docs.append(doc)

        unique_sources = list(set(all_sources))
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        print("Unique Sources:", unique_sources)
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

        unique_sources = prioritize_sources(original_message, unique_sources)
        return {"sources_url": unique_sources, "sources_documents": all_docs}
     
 
    @staticmethod
    @tool
    def get_custom_date_range_articles(start_date: str, end_date: str = None, article_type: str = "all", language_code: str = 'en') -> str:
        """
        Retrieves articles published within a specific date range.
        
        **Use When:**
        - The user specifies a date range or single date for searching.
        - Example Queries: "Articles from January 2023", "Find articles between March and April."
        
        **Parameters:**
        - start_date: The start of the date range in YYYY-MM-DD format.
        - end_date: (Optional) The end of the date range in YYYY-MM-DD format.
        - article_type: <fact-check/law/explainers/decode/mediabuddhi/web-stories/boom-research/deepfake-tracker/all>
        Provide one keyword from the list if present in the query or related to any word in the query. If not related to any specific type, return 'all'. Note: If query has "boom-report" then use "boom-research".
        - language_code: A short code for the language of the query. Supported values are:
                - "en" for English
                - "hi" for Hindi
                - "bn" for Bangla
        **Returns:**
        - Articles published within the specified date range.
        """
        print("Inside get_custom_date_range_articles")

        sources = fetch_custom_range_articles_urls(start_date, end_date, article_type, language_code)
        
        # Validate date range
        return {"sources_url": sources}
    

    @staticmethod
    @tool
    def get_latest_articles(article_type: str, language_code: str = 'en') -> str:
        """
        Use this tool when you need to retrieve the most recent articles.
        Essential for staying updated with the newest content and latest developments.
        **Use When:**
        - The user asks for the latest news.
        - Example Queries: "Provide Latest Articles", "Provide latest factchecks", "Provide latest explianers"
        Parameters:
        - article_type: <fact-check/law/explainers/decode/mediabuddhi/web-stories/boom-research/deepfake-tracker/scamcheck/all>
        Provide one keyword from the list if present in the query or related to any word in the query. If not related to any specific type, return 'all'. Note: If query has "boom-report" then use "boom-research".
        - language_code: A short code for the language of the query. Supported values are:
                - "en" for English
                - "hi" for Hindi
                - "bn" for Bangla
        Returns:
        - List of the most recent articles within the specified time window
        """
        print("Inside get_latest_articles")
        sources = fetch_latest_article_urls(article_type, language_code)

        return {"sources_url": sources}


    @staticmethod
    @tool
    def get_articles_by_topic(topic: str, language_code: str = 'en') -> str:
        """
        Retrieves articles specifically focused on a particular topic or subject.

        **Use When:**
        - The user explicitly mentions a subject or category they want information on.
        - Example Queries: "Fact check on Kumbh Mela", "Articles about LLM architecture."

        **Parameters:**
        - topic: A string specifying the subject area.
        - language_code: A short code for the language of the query. Supported values are:
                        - "en" for English
                        - "hi" for Hindi
                        - "bn" for Bangla
        **Returns:**
        - Articles related to the specified topic.
        """
        if language_code == 'hi':
            domain = 'https://hi.boomlive.in'
        elif language_code == 'bn':
            domain = 'https://bn.boomlive.in'
        else:
            domain = 'https://www.boomlive.in'

        topic_url =  f"{domain}/search?search={topic}"
        sources = extract_articles(topic_url,language_code) 
        sources_url = [source[1] for source in sources] 

        return  {"sources": sources, "sources_url": sources_url}
  