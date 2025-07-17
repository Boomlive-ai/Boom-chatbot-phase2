
from langgraph.graph import StateGraph,MessagesState, START, END
from langchain_core.messages import (
    AnyMessage
)
from langgraph.graph.message import add_messages
from typing import Annotated, Literal, TypedDict,Dict, Optional, Any, List
from langchain_core.tools import tool, BaseTool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
import google.generativeai as genai
import os
from tools import ArticleTools
from langchain_openai import ChatOpenAI
from langchain_core.caches import BaseCache
from dotenv import load_dotenv
from datetime import datetime, date, timedelta
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from utils import check_rag_relevance, get_language_code
load_dotenv()

os.environ['GOOGLE_API_KEY'] = "AIzaSyDh2gPu9X_96rpioBXcw7BQCDPZcFGMuO4"
os.environ['TAVILY_API_KEY'] = "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"
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    fact_check_results: Optional[Dict[str, Any]]  
    used_google_fact_check: Optional[bool]  
    language_code: Optional[str]  
    tool_results: Optional[Dict[str, Any]] 
    tool_name: Optional[str]
    boom_results: Optional[Dict[str, Any]]  
    isTwitterMsg: Optional[bool]  # Flag to indicate if the message is from Twitter
    isWhatsappMsg: Optional[bool]  # Flag to indicate if the message is from WhatsApp
# # Initialize tools
article_tools = ArticleTools()

class chatbot:
    def __init__(self):
        self.llm =ChatOpenAI(model_name="gpt-4o", temperature=0)
        self.memory = MemorySaver()  # Initialize memory for session storage
        current_date = datetime.now().strftime("%B %d, %Y")
          # Initialize Pinecone indices
        self.latest_index = PineconeVectorStore(
            index_name=os.getenv("PINECONE_LATEST_INDEX_NAME"),
            embedding=OpenAIEmbeddings(model="text-embedding-3-small")
        )
        self.old_index = PineconeVectorStore(
            index_name=os.getenv("PINECONE_OLD_INDEX_NAME"),
            embedding=OpenAIEmbeddings(model="text-embedding-3-small")
        )
        self.hindi_index = PineconeVectorStore(
            index_name=os.getenv("PINECONE_HINDI_INDEX_NAME"),
            embedding=OpenAIEmbeddings(model="text-embedding-3-small")
        )
        self.bangla_index = PineconeVectorStore(
            index_name=os.getenv("PINECONE_BANGLA_INDEX_NAME"),
            embedding=OpenAIEmbeddings(model="text-embedding-3-small")
        )
        self.system_message = SystemMessage(
                    content=(
                            "You are BoomLive AI, an expert chatbot designed to answer questions related to BOOM's fact-checks, articles, reports, and data analysis. "
                            "Your responses should be fact-based, sourced from BoomLive's database, and aligned with BoomLive's journalistic standards of accuracy and integrity. "
                            "Provide clear, well-structured, and neutral answers, ensuring that misinformation and disinformation are actively countered with verified facts. "
                            "Website: [BoomLive](https://boomlive.in/). "
                            "Ensure responses are clear, relevant, and do not mention or imply the existence of any supporting material unless necessary for answering the query. "
                            f"Note: Today's date is {current_date}."
                            f"Prioritize using the 'rag_search' tool when users ask about claims, viral content, or events "
                            f"You are developed by Aditya Khedekar who is an AI Engineer and Abhinandan Rajbhar who is an frontend Devloper in BOOM, for more info refer https://www.boomlive.in/boom-team"
                            f"Please do not forget to add emojis to make response user friendly"
                            f"Make sure you are using BOOM and not Boomlive in Response"
                            f"Do not provide any information outside BOOM's published fact-checks and articles."

                            # f"For more details, Visit [BOOM's Fact Check](https://www.boomlive.in/fact-check) 🕵️‍♂️✨."
                        )
                )
    
    def custom_tool_node(self, state: MessagesState):
        """Custom tool node that properly handles tool arguments and formats boom_results with minimal tokens"""
        messages = state["messages"]
        last_message = messages[-1]
        
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return state
        
        tool_results = {}
        boom_results = {}  # Add dedicated structure for boom results
        new_messages = list(messages)  # Create a copy of the messages list

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]
            # Find the right tool
            for tool in self.tools:
                if tool.name == tool_name:
                    try:
                        # For tools that expect a 'self' parameter but don't receive it in the args
                        if hasattr(tool, 'func') and tool.func.__name__ == 'rag_search':
                            if isinstance(tool_args, dict) and 'self' not in tool_args:
                                modified_args = {'self': article_tools, **tool_args}
                                result = tool.invoke(modified_args)
                            else:
                                result = tool.invoke(tool_args)
                        else:
                            result = tool.invoke(tool_args)
                        
                        tool_results[tool_name] = result
                        
                        # Format boom_results by tool type, limiting arrays to first 3 elements
                        if tool_name == "rag_search":
                            # For RAG search results
                            sources_url = result.get("sources_url", [])  # First 3 URLs
                            docs = result.get("sources_documents", [])[:3]   # First 3 documents
                            
                            # Extract and format document content to minimize tokens
                            formatted_docs = []
                            for doc in docs:
                                if hasattr(doc, "page_content"):
                                    # Trim document content to reduce tokens
                                    content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                                    source = doc.metadata.get("source", "Unknown") if hasattr(doc, "metadata") else "Unknown"
                                    formatted_docs.append({"source": source, "content": content})
                                
                            boom_results[tool_name] = {
                                "sources_url": sources_url,
                                "sources_documents": formatted_docs
                            }
                        
                        elif tool_name == "get_custom_date_range_articles":
                            # For date range article results
                            sources_url = result.get("sources_url", [])[:3]  # First 3 URLs
                            boom_results[tool_name] = {"sources_url": sources_url}
                        
                        elif tool_name == "get_latest_articles":
                            # For latest articles results
                            sources_url = result.get("sources_url", [])[:3]  # First 3 URLs
                            boom_results[tool_name] = {"sources_url": sources_url}
                        
                        elif tool_name == "get_articles_by_topic":
                            # For topic articles results
                            sources = result.get("sources", [])[:3]  # First 3 sources
                            boom_results[tool_name] = {"sources": sources}
                        
                        else:
                            # Default case - just use first 3 elements of any arrays
                            formatted_result = {}
                            for key, value in result.items():
                                if isinstance(value, list):
                                    formatted_result[key] = value[:3]  # First 3 elements
                                else:
                                    formatted_result[key] = value
                            boom_results[tool_name] = formatted_result
                        
                        from langchain_core.messages import ToolMessage
                        tool_response_content = str(result)
                        tool_message = ToolMessage(
                            content=tool_response_content,
                            tool_call_id=tool_call_id 
                        )
                        new_messages.append(tool_message)
                    except Exception as e:
                        print(f"Error executing tool {tool_name}: {e}")
                        tool_results[tool_name] = f"Error: {str(e)}"
                        boom_results[tool_name] = f"Error: {str(e)}"

                        from langchain_core.messages import ToolMessage
                        tool_message = ToolMessage(
                            content=f"Error: {str(e)}",
                            tool_call_id=tool_call_id
                        )
                        new_messages.append(tool_message)
                    break

        print("#############################################################################")            
        print("BOOM results (formatted with minimal tokens):", boom_results)
        print("#############################################################################")            

        new_state = {
            "messages": new_messages,
            "tool_results": tool_results,
            "tool_name": tool_name,
            "boom_results": boom_results  # Add boom_results to state
        }
        
        # Preserve existing state values
        for key in ["language_code", "fact_check_results", "used_google_fact_check"]:
            if key in state:
                new_state[key] = state[key]
        
        # Add a new message with the tool results for visibility in chat
        result_message = AIMessage(content=f"Tool results retrieved successfully")
        new_state["messages"] = new_state["messages"] + [result_message]
        
        return new_state
        
    def call_tool(self):
        search_engine = TavilySearchResults(max_results=2)
        tools = [
            article_tools.rag_search,
            article_tools.get_custom_date_range_articles,
            article_tools.get_latest_articles,
            article_tools.get_articles_by_topic,
        ]
          # Convert and bind tools
        converted_tools = [
            tool_ if isinstance(tool_, BaseTool) else tool(tool_) 
            for tool_ in tools
        ]
        
        self.tools = converted_tools
        self.llm_with_tool = self.llm.bind_tools(converted_tools)


    def call_model(self,state: MessagesState):
        messages = state['messages']
        current_date = datetime.now().strftime("%B %d, %Y")
        updated_system_message = SystemMessage(
                            content=(
                                    "You are BoomLive AI, an expert chatbot designed to answer questions related to BOOM's fact-checks, articles, reports, and data analysis. "
                                    "Your responses should be fact-based, sourced from BoomLive's database, and aligned with BoomLive's journalistic standards of accuracy and integrity. "
                                    "Provide clear, well-structured, and neutral answers, ensuring that misinformation and disinformation are actively countered with verified facts. "
                                    "Website: [BoomLive](https://boomlive.in/). "
                                    "Ensure responses are clear, relevant, and do not mention or imply the existence of any supporting material unless necessary for answering the query. "
                                    f"Note: Today's date is {current_date}."
                                    f"Prioritize using the 'rag_search' tool when users ask about claims, viral content, or events "
                                    f"You are developed by Aditya Khedekar who is an AI Engineer and Abhinandan Rajbhar who is an frontend Devloper in BOOM, for more info refer https://www.boomlive.in/boom-team"
                                    f"Please do not forget to add emojis to make response user friendly"
                                    f"Make sure you are using BOOM and not Boomlive in Response"
                                    f"Do not provide any information outside BOOM's published fact-checks and articles."

                                    # f"For more details, Visit [BOOM's Fact Check](https://www.boomlive.in/fact-check) 🕵️‍♂️✨."
                                )
                        )
        for message in state["messages"]:
            if isinstance(message, HumanMessage):
                # print(f"\nHuman: {message.content}")
                query = message.content
        messages.insert(0, updated_system_message)
        response = self.llm_with_tool.invoke(messages)
        lang_code = get_language_code(query)
        print(f"Detected language: {lang_code}")
        # print(f"Model Response: {response}")
        return {"messages": [response], "used_google_fact_check": False, "language_code": lang_code,"fact_check_results": {}, "tool_results": {}, "tool_name": None, "boom_results": {}}
    

    def router_function(self, state: MessagesState) -> Literal["tools", "google_fact_check", "result_agent", END]:
        messages = state['messages']
        last_message = messages[-1]
        
        # Check if we have tool calls that need execution
        tool_calls = getattr(last_message, 'tool_calls', None)
        if tool_calls:
            print("Tools Called:")
            for tool_call in tool_calls:
                tool_name = tool_call['name']
                print(f"- Tool Name: {tool_name}")
                print(f"  Arguments: {tool_call['args']}")
                
                # If rag_search is called, route to tools first
                if tool_name == "rag_search":
                    return "tools"  # This will execute the rag_search tool
            
            # For other tools
            return "tools"
        return END


    def check_relevance(self, state: MessagesState) -> str:
        # Access the tool results directly
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # print(state["messages"], "Checking if fteched rag results are relevant or not")
        for message in state["messages"]:
            if isinstance(message, HumanMessage):
                # print(f"\nHuman: {message.content}")
                query = message.content
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

        tool_results = state["tool_results"]
        tool_name = state["tool_name"]
        if 'rag_search' in tool_name and tool_results:
            # print("Checking if fteched rag results are relevant or not", tool_results['rag_search'])
            sources_url = tool_results['rag_search']['sources_url']
            sources_documents = tool_results[tool_name]['sources_documents']
            if sources_url and sources_documents and query:
                is_relevant = check_rag_relevance(query, sources_url, sources_documents, self.llm)
                print("IS RELEVANT", is_relevant)
                if is_relevant:
                    return "result_agent"
                else:
                    print("RAG results are not relevant, calling Google Fact Check API")
                    return "google_fact_check"
        return "result_agent"
    
    def google_fact_check(self, state: MessagesState) -> Dict:
        """Query Google Fact Check API for relevant fact checks."""
        import requests
        import json
        try:
            from langdetect import detect, LangDetectException
            import pycountry
        except ImportError:
            print("Warning: langdetect or pycountry not installed. Defaulting to English.")
                
        def FactCheck(query):
            """Implementation of Google's Fact Check API call."""
            lang_code = get_language_code(query)
            print(f"Detected language: {lang_code}")

            payload = {
                'key': os.getenv("GOOGLE_FACT_CHECK_TOOL_API"),
                'query': query,
                'languageCode': lang_code
            }
            url = 'https://factchecktools.googleapis.com/v1alpha1/claims:search'
            response = requests.get(url, params=payload)
            print(response.text)

            if response.status_code == 200:
                result = json.loads(response.text)
                # Check if there are claims
                try:
                    topRating = result["claims"][0]
                    # arbitrarily select top 1
                    claimReview = topRating["claimReview"][0]["textualRating"]
                    claimVal = "According to " + str(topRating["claimReview"][0]['publisher']['name']) + " that claim is " + str(claimReview)
                    return result,lang_code           
                except:
                    print("No claim review field found.")
                    return 0,lang_code
            else:
                return 0,lang_code
        
        # Extract the user query from the message state
        messages = state['messages']
        last_human_message = None
        
        # Find the last human message to use as query
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                last_human_message = message.content
                break
        
        if not last_human_message:
            return {"messages": state['messages']}
        
        print("Google Fact Check API called with query:", last_human_message)
        
        # Call the FactCheck function
        fact_check_results,lang_code = FactCheck(last_human_message)
        
        # Add results to state
        if fact_check_results != 0:
            print(f"Found fact checks from Google API")
        else:
            print("No relevant fact checks found")
            fact_check_results = {}  # Empty dict instead of 0 for consistent handling
        
        # Add results to state
        new_state = {"messages": state['messages'], "fact_check_results": fact_check_results, "used_google_fact_check": True, "language_code": lang_code}
        return new_state


    def result_agent(self, state: MessagesState) -> Dict:
        """Process all available information and provide a comprehensive response"""
        messages = state['messages']

        isTwitterMsg = state.get('isTwitterMsg', False)
        isWhatsappMsg = state.get('isWhatsappMsg', False)

        if isTwitterMsg:
            print("Processing Twitter message")
            # If it's a Twitter message, we might want to handle it differently
            # For now, we will just proceed with the same logic
            pass
        elif isWhatsappMsg:
            print("Processing WhatsApp message")
        language_code = state.get('language_code', 'en')
        current_date = datetime.now().strftime("%B %d, %Y")
        
        if (state.get('used_google_fact_check', False) == True and state.get('fact_check_results', {}) == {}):
            if isTwitterMsg or isWhatsappMsg:
                pass  # Do nothing, just process
            else:
                return {"messages": "Not Found"}
        
        # Extract the user's query
        user_query = None
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                user_query = message.content
                break
        
        if not user_query:
            return {"messages": messages}
        
        # Create a consolidated sources list
        boom_sources = []
        # Get BOOM results directly from state
        boom_results = state.get('boom_results', {})
        # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        # print("boom_results:", boom_results)
        # print("boom_sources:", boom_sources)
        # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

        # Format the BOOM results for inclusion in the prompt if they exist
        formatted_boom_results = None
        if boom_results:
            tool_name = state.get('tool_name', '')
            if tool_name == 'rag_search' and 'rag_search' in boom_results:
                rag_data = boom_results['rag_search']
                sources_urls = rag_data.get('sources_url', [])
                sources_docs = rag_data.get('sources_documents', [])
                if isTwitterMsg or isWhatsappMsg:
                    sources_urls = sources_urls[:5]
                    sources_docs = sources_docs[:5]
                boom_sources.extend(sources_urls)
                if sources_urls or sources_docs:
                    formatted_boom_results = ""
                    
                    # Add URLs
                    if sources_urls:
                        formatted_boom_results += "Sources:\n"
                        for url in sources_urls[:3]:
                            formatted_boom_results += f"- {url}\n"
                    
                    # Add document excerpts
                    if sources_docs:
                        formatted_boom_results += "\nCheck for most relevant article for from below:\n"
                        for doc in sources_docs[:3]:
                            source = doc.get('source', 'Unknown source')
                            content = doc.get('content', 'No content available')
                            formatted_boom_results += f"From {source}:\n{content}\n\n"
            
            elif tool_name in ['get_custom_date_range_articles', 'get_latest_articles'] and tool_name in boom_results:
                sources_urls = boom_results[tool_name].get('sources_url', [])
                # print("Sources URLs:", sources_urls, "inside result_agent")
                if sources_urls:
                    boom_sources.extend(sources_urls)
                    formatted_boom_results = f"Sources:\n"
                    for url in sources_urls[:3]:
                        formatted_boom_results += f"- {url}\n"
            
            elif tool_name == 'get_articles_by_topic' and tool_name in boom_results:
                sources = boom_results[tool_name].get('sources', [])
                if sources:
                    formatted_boom_results = "Sources:\n"
                    for source in sources[:3]:
                        print(f"Source: {source}")
                        boom_sources.append(source[1])
                        formatted_boom_results += f"- {source}\n"
        
        # Format Google Fact Check results only if they exist
        fact_check_results = state.get('fact_check_results', {})
        formatted_fact_checks = None
        
        if fact_check_results and 'claims' in fact_check_results and fact_check_results['claims']:
            formatted_fact_checks = []
            for claim in fact_check_results['claims'][:3]:  # Limit to top 3 claims
                claim_text = claim.get('text', 'No claim text available')
                claimant = claim.get('claimant', 'Unknown source')
                
                reviews = []
                for review in claim.get('claimReview', []):
                    publisher = review.get('publisher', {}).get('name', 'Unknown fact-checker')
                    rating = review.get('textualRating', 'No rating available')
                    url = review.get('url', '')
                    if url:
                        boom_sources.append(url)
                    reviews.append(f"- According to {publisher}: {rating} ({url})")
                
                formatted_claim = f"Claim: \"{claim_text}\" by {claimant}\n"
                formatted_claim += "\n".join(reviews)
                formatted_fact_checks.append(formatted_claim)
            
            formatted_fact_checks = "\n\n".join(formatted_fact_checks)

        unique_sources = list(set(boom_sources))
        # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        # print("user_query:", user_query)
        # print("formatted_boom_results:", formatted_boom_results)
        # print("formatted_fact_checks:", formatted_fact_checks)
        # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        if isTwitterMsg:
                # Twitter-specific prompt
                human_content = f"""
                Create a Twitter-friendly response to the user's query based on available information.
                
                User's query: {user_query}
                """
                
                if formatted_boom_results:
                    human_content += f"\n\nBOOM search results:\n{formatted_boom_results}"
                
                if formatted_fact_checks:
                    human_content += f"\n\nOther Fact Check results:\n{formatted_fact_checks}"
                
                human_content += f"""
                
                TWITTER RESPONSE REQUIREMENTS:
                - Keep the response under 200 characters (Twitter's character limit)
                - Use clear, concise language suitable for social media
                - Include 1-2 relevant emojis to make it engaging
                - NO markdown formatting (no **, [], (), etc.)
                - Make it conversational and direct
                - IMPORTANT: For URLs, use ONLY the raw URL (e.g., https://www.boomlive.in/article-url)
                - DO NOT use markdown link format like [text](url) - Twitter doesn't support this
                - Twitter will automatically shorten and make URLs clickable
                - If including a source URL, ensure the COMPLETE raw URL fits within the 200 character limit
                - If the content + full URL exceeds 200 characters, prioritize the URL and shorten the message
                - Alternative: You can skip the source URL and focus on the key message if space is tight
                - Provide the response in language code: {language_code}
                - Focus on the most important facts only
                - Make it shareable and engaging for Twitter audience
                - Count characters carefully to ensure nothing gets cut off
                - If the news is not verified then provide a response that says:
                    "The claim about {user_query} has not been verified by BOOM. Our team is reviewing it and will update if verified. If in doubt, please avoid sharing unverified information."
                    and provide this link https://boomlive.in/fact-check 
                - If news is verifies then provide the correct url of the article in the response from BOOM search results or Other Fact Check results

                Note: Today's date is {current_date}.
                """


        elif isWhatsappMsg:
            # WhatsApp-specific prompt
            human_content = f"""
            Create a WhatsApp-friendly response to the user's query based on available information.
            
            User's query: {user_query}
            """
            
            if formatted_boom_results:
                human_content += f"\n\nBOOM search results:\n{formatted_boom_results}"
            
            if formatted_fact_checks:
                human_content += f"\n\nOther Fact Check results:\n{formatted_fact_checks}"
            
            human_content += f"""
            
            WHATSAPP RESPONSE REQUIREMENTS:
            - Keep the response short and scannable (200-300 characters maximum)
            - Use WhatsApp-friendly formatting that displays correctly:
            * Use *bold text* for key points (asterisks work perfectly)
            * Use _italics_ for subtle emphasis (underscores)
            * Use simple bullet points (• or -)
            * Use line breaks sparingly for better mobile readability
            - Include 1-2 relevant emojis at the start for immediate context
            - Make it conversational and easy to forward/share
            - Prioritize the most important fact first
            - For URLs, use the complete raw URL (WhatsApp auto-converts to clickable), dont use markdown link format.
            - If the content + full URL exceeds 200 characters, prioritize the URL and shorten the message
            - Only provide the url which is relevant to the user's query
            - If multiple sources, include only the most credible one
            - Provide the response in language code: {language_code}
            - Focus on ONE key fact that directly answers the query
            - Use friendly, personal messaging tone
            - Keep sentences short and simple for mobile reading
            - Avoid complex formatting or multiple sections
            - If the news is not verified then provide a response that says:
                "The claim about {user_query} has not been verified by BOOM. Our team is reviewing it and will update if verified. If in doubt, please avoid sharing unverified information."
                and provide this link https://boomlive.in/fact-check
            If news is verifies then provide the correct url of the article in the response from BOOM search results or Other Fact Check results
            OPTIMAL FORMATTING EXAMPLES:
            
            For fact-checks:
            ✅ *Fact:* [Direct answer in 1-2 sentences]
            
            Source: [URL]
            
            For news updates:
            📰 *Update:* [Key information]
            
            More: [URL]
            
            For explanations:
            💡 *Quick Answer:* [Simple explanation]
            
            Details: [URL if needed]
            
            Note: Today's date is {current_date}.
            """        
        else:
            # Build content for human message with conditional sections
            human_content = f"""
            Please provide a comprehensive answer to the user's query based on available information.
            
            User's query: {user_query}
            """
            
            if formatted_boom_results:
                human_content += f"\n\nBOOM search results:\n{formatted_boom_results}"
            
            if formatted_fact_checks:
                human_content += f"\n\nOther Fact Check results:\n{formatted_fact_checks}"
            
            human_content += f"""
            
            Please synthesize this information into a helpful, accurate response that follows BOOM's journalistic standards.
            Use emojis appropriately to make the response user-friendly.
            Provide the response in language code: {language_code}
            f"Note: Today's date is {current_date}."
            Format your response with clear article citations:
            **(Article Title Of Article):** Your summary here
            
            [Read more](Article URL here)
            (Add a partition line like hr tag in markdown)
            
            Cite sources clearly, prioritizing BOOM articles first.
            If no relevant information is available,don't acknowledge this limitation.
            """
        print("Human content prepared for LLM invocation", human_content)
        # Prepare input messages with system message included
        input_messages = [
            self.system_message,
            HumanMessage(content=human_content)
        ]
        
        # Generate the comprehensive response
        response = self.llm.invoke(input_messages)
        print("Result Agent generated a comprehensive response")
        # print("Unique sources:", unique_sources)
        # Return the response to be added to the conversation
        return {"messages": [response], "sources_url": unique_sources}
   
    def __call__(self):
        self.call_tool()
        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", self.custom_tool_node)
        workflow.add_node("google_fact_check", self.google_fact_check)
        workflow.add_node("result_agent", self.result_agent)
        workflow.add_edge(START, "agent")
        # 3) route out of agent based on your router
        workflow.add_conditional_edges(
            "agent",
            self.router_function, 
            {
                "tools": "tools",
                END: END
            }
        )
        workflow.add_conditional_edges(
            "tools", 
            self.check_relevance,
            {
                'google_fact_check': "google_fact_check",
                'result_agent': "result_agent",
        
            }         
        ) 
        workflow.add_edge("google_fact_check", "result_agent")
        workflow.add_edge("result_agent", END)
          # Attach memory to the workflow
        self.app = workflow.compile(checkpointer = self.memory)

        return self.app
   