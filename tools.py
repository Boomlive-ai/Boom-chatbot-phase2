from langchain_core.tools import tool, InjectedToolArg
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from datetime import datetime, timedelta
import re, os
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from utils import fetch_custom_range_articles_urls, fetch_latest_article_urls,extract_articles

class ArticleTools:
    def __init__(self):
        pass
    @staticmethod
    @tool
    def rag_search(query: str, language_code: str) -> str:
        """
        This tool performs semantic searches across articles to find relevant information that matches the user's query based on meaning rather than keywords.

        **Use When:** 
        - The user is asking about a specific claim, event, or incident (e.g., "Modi chants in the US", or descriptive queries like "7 साल पुराना वीडियो").
        - The query suggests the need for fact-checking or detailed analysis of an occurrence or statement.
        - Broad or open-ended questions where semantic understanding is required (e.g., "What are RAG systems?", "Explain Retrieval Augmented Generation").

        **Parameters:**
        - query: A string representing the user's search query without any changes in the query.
        - language_code: A short code for the language of the query. Supported values are:
                - "en" for English
                - "hi" for Hindi
                - "bn" for Bangla

        **Returns:**
        - Article content semantically related to the query, providing relevant information or fact-based insights.
        """
        print("Inside rag_search",query, language_code)
        latest_index = PineconeVectorStore(
            index_name=os.getenv("PINECONE_LATEST_INDEX_NAME"),
            embedding=OpenAIEmbeddings(model="text-embedding-3-small")
        )
        old_index = PineconeVectorStore(
            index_name=os.getenv("PINECONE_OLD_INDEX_NAME"),
            embedding=OpenAIEmbeddings(model="text-embedding-3-small")
        )
        hindi_index = PineconeVectorStore(
            index_name=os.getenv("PINECONE_HINDI_INDEX_NAME"),
            embedding=OpenAIEmbeddings(model="text-embedding-3-small")
        )
        bangla_index = PineconeVectorStore(
            index_name=os.getenv("PINECONE_BANGLA_INDEX_NAME"),
            embedding=OpenAIEmbeddings(model="text-embedding-3-small")
        )
        index_to_use = "both"
        if language_code=='hi':
            index_to_use='hindi-boom-articles'
        elif language_code=='bn':
            index_to_use='bangla-boom-articles'
        
        all_docs = []
        all_sources = []
    
        # Simple mock implementation of semantic search
        # Clean up index_to_use if needed
        if index_to_use is not None:
            index_to_use = index_to_use.split(".")[-1].strip()
        
        # Retrieve documents based on selected index
        if index_to_use in ["latest", None, "both"]:
            latest_retriever = latest_index.as_retriever(search_kwargs={"k": 5})
            latest_docs = latest_retriever.get_relevant_documents(query)
            all_docs.extend(latest_docs)
            all_sources.extend([doc.metadata.get("source", "Unknown") for doc in latest_docs])
            print(f"Latest documents retrieved: {len(latest_docs)}")
        
        if index_to_use in ["old", "both"]:
            old_retriever = old_index.as_retriever(search_kwargs={"k": 5})
            old_docs = old_retriever.get_relevant_documents(query)
            all_docs.extend(old_docs)
            all_sources.extend([doc.metadata.get("source", "Unknown") for doc in old_docs])
            print(f"Old documents retrieved: {len(old_docs)}")
        
        if index_to_use == "hindi-boom-articles":
            hindi_retriever = hindi_index.as_retriever(search_kwargs={"k": 5})
            hindi_docs = hindi_retriever.get_relevant_documents(query)
            all_docs.extend(hindi_docs)
            all_sources.extend([doc.metadata.get("source", "Unknown") for doc in hindi_docs])
            print(f"Hindi documents retrieved: {len(hindi_docs)}")
        
        if index_to_use == "bangla-boom-articles":
            bangla_retriever = bangla_index.as_retriever(search_kwargs={"k": 5})
            bangla_docs = bangla_retriever.get_relevant_documents(query)
            all_docs.extend(bangla_docs)
            all_sources.extend([doc.metadata.get("source", "Unknown") for doc in bangla_docs])
            print(f"Bangla documents retrieved: {len(bangla_docs)}")

        unique_sources = list(set(all_sources))

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
        - article_type: <fact-check/law/explainers/decode/mediabuddhi/web-stories/boom-research/deepfake-tracker/all>
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
  