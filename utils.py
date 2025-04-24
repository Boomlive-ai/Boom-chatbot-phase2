import datetime, requests, re
from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect, LangDetectException

    
def get_language_code(text):
    """
    Detects language and returns ISO 639-1 two-letter language code.
    Falls back to 'en' if not Hindi ('hi') or Bangla ('bn').
    Args:
        text (str): Text to detect language from
    Returns:
        str: 'hi', 'bn', or default 'en'
    """
    try:
        detected_code = detect(text)
        # Validate code
        if detected_code in ['hi', 'bn']:
            return detected_code
        else:
            return 'en'
    except (LangDetectException, NameError):
        return 'en'
        
def validate_date_range(from_date: str, to_date: str) -> bool:
    """
    Validate the custom date range.

    Args:
        from_date (str): Start date.
        to_date (str): End date.

    Returns:
        bool: True if valid, False otherwise.
    """
    try:
        from_dt = datetime.datetime.strptime(from_date, '%Y-%m-%d')
        to_dt = datetime.datetime.strptime(to_date, '%Y-%m-%d')
        return from_dt <= to_dt
    except ValueError:
        return False
    



def fetch_custom_range_articles_urls(from_date: str = None, to_date: str = None, article_type: str = "all", language_code: str = "en"): 
    """
    Fetch and return article URLs based on a custom date range.

    Args:
        from_date (str): Start date in 'YYYY-MM-DD' format. Defaults to 6 months ago.
        to_date (str): End date in 'YYYY-MM-DD' format. Defaults to today.

    Returns:
        list: List of article URLs.
    """
    # Initialize variables
    article_urls = []
    start_index = 0
    count = 20
    if language_code == 'hi':
        api_domain = 'https://hindi.boomlive.in'
        sid = 'A2mzzjG2Xnru2M0YC1swJq6s0MUYXVwJ4EpJOub0c2Y8Xm96d26cNrEkAyrizEBD'
    elif language_code == 'bn':
        api_domain = 'https://bangla.boomlive.in'
        sid = 'xgjDMdW01R2vQpLH7lsKMb0SB5pDCKhFj7YgnNymTKvWLSgOvIWhxJgBh7153Mbf'
    else:
        api_domain = 'https://www.boomlive.in'
        sid = '1w3OEaLmf4lfyBxDl9ZrLPjVbSfKxQ4wQ6MynGpyv1ptdtQ0FcIXfjURSMRPwk1o'

    excluded_types = ["fact-check", "law", "explainers", "decode", "mediabuddhi", 
                 "web-stories", "boom-research", "deepfake-tracker","news","fast-check","partner-content","weekly-wrap","boom-reports","videos"]

    print("fetch_custom_range_articles_urls", from_date, to_date, article_type)
    # Calculate default date range if not provided
    current_date = datetime.date.today()
    if not to_date:
        to_date = current_date.strftime('%Y-%m-%d')
    if not from_date:
        custom_months_ago = current_date - datetime.timedelta(days=180)  # Default to 6 months ago
        from_date = custom_months_ago.strftime('%Y-%m-%d')

    # Validate the date range
    if not validate_date_range(from_date, to_date):
        print("Invalid date range. Ensure 'from_date' <= 'to_date' and format is YYYY-MM-DD.")
        return []

    print(f"Fetching article URLs from {from_date} to {to_date}....")

    # Loop to fetch article URLs in batches
    while True:
        print("Current start index:", start_index)

        # Construct API URL with the custom range
        api_url = f'{api_domain}/dev/h-api/news?startIndex={start_index}&count={count}&fromDate={from_date}&toDate={to_date}'
        headers = {
            "accept": "*/*",
            "s-id": sid
        }
        print(f"Requesting API URL: {api_url}")

        # Make the API request
        response = requests.get(api_url, headers=headers)
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()

            # Break the loop if no articles are returned
            if not data.get("news"):
                break

            # Extract article URLs from the response
            for news_item in data.get("news", []):
                url_path = news_item.get("url")
                if url_path:
                    if article_type == "all":
                        article_urls.append(url_path)  # Include all URLs
                    elif url_path and f"{api_domain}/{article_type}" in url_path:
                        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                        print(url_path)
                        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

                        article_urls.append(url_path)
                    else:
                        if url_path and not any(f"{api_domain}/{excluded}" in url_path for excluded in excluded_types):
                            article_urls.append(url_path)

            start_index += count
        else:
            print(f"Failed to fetch articles. Status code: {response.status_code}")
            break
    print("article_urls",article_urls)        
    return article_urls



def fetch_latest_article_urls(article_type: str, language_code: str = "en"):
    """
    Fetches the latest articles from the BoomLive API, filters them based on exact keyword matching
    (fact-check, decode, explainers, mediabuddhi, boom-research), and sorts them by the largest number at the end of the URL.
    Then it returns the top 5 filtered URLs.

    Args:
        query (str): The query string to filter the articles.

    Returns:
        list: A list of the top 5 filtered URLs, sorted by the largest number at the end of the URL.
    """

    urls = []
 # Determine correct API base URL based on language_code
    if language_code == 'hi':
        api_domain = 'https://hindi.boomlive.in'
        sid = 'A2mzzjG2Xnru2M0YC1swJq6s0MUYXVwJ4EpJOub0c2Y8Xm96d26cNrEkAyrizEBD'
    elif language_code == 'bn':
        api_domain = 'https://bangla.boomlive.in'
        sid = 'xgjDMdW01R2vQpLH7lsKMb0SB5pDCKhFj7YgnNymTKvWLSgOvIWhxJgBh7153Mbf'
    else:
        api_domain = 'https://www.boomlive.in'
        sid = '1w3OEaLmf4lfyBxDl9ZrLPjVbSfKxQ4wQ6MynGpyv1ptdtQ0FcIXfjURSMRPwk1o'

    # Construct API endpoint
    api_url = f"{api_domain}/dev/h-api/news"

    headers = {
        "accept": "*/*",
        "s-id": sid
    }


    print(f"Fetching articles from API: {api_url}")

    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes
        
        if response.status_code == 200:
            data = response.json()

            # Break if no articles are found
            if not data.get("news"):
                return []

            for news_item in data.get("news", []):
                url_path = news_item.get("url")
                
                if article_type == "all":
                    urls.append(url_path)  # Include all URLs
                elif url_path and f"{api_domain}/{article_type}" in url_path:
                    urls.append(url_path)

    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch articles: {e}")
        return {"error": f"Failed to fetch articles: {e}"}

    # Extract numeric values from URLs and sort by the largest number at the end of the URL
    urls_with_numbers = []

    for url in urls:
        # Extract the number at the end of the URL
        match = re.search(r'(\d+)(?=\s*$)', url)
        if match:
            number = int(match.group(0))
            urls_with_numbers.append((url, number))
    
    # Sort by the numeric values in descending order (largest number first)
    sorted_urls = sorted(urls_with_numbers, key=lambda x: x[1], reverse=True)

    # Get the top 5 filtered URLs
    top_5_urls = [url for url, _ in sorted_urls[:5]]

    print(f"Top 5 filtered URLs: {top_5_urls}")
    return top_5_urls




def extract_articles(tag_url, language_code):
    """Fetch and extract article titles, URLs, and summaries from BoomLive search results."""
    if language_code == 'hi':
        BASE_URL = 'https://hindi.boomlive.in'
    elif language_code == 'bn':
        BASE_URL = 'https://bangla.boomlive.in'
    else:
        BASE_URL = 'https://www.boomlive.in'
    try:
        response = requests.get(tag_url, timeout=10)
        if response.status_code != 200:
            print("Failed to retrieve page, status code:", response.status_code)
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = []
        
        # Select all <a> tags with class "heading_link" inside the section with class "search-page"
        for link in soup.select("section.search-page a.heading_link"):
            title = link.get_text(strip=True)
            url = link.get("href")
            # Ensure full URL if the link is relative
            if url and not url.startswith("http"):
                url = f"{BASE_URL}{url}"
            
            # Find the closest parent <h4> and then the next sibling <p> for summary text
            h4_tag = link.find_parent("h4")
            if h4_tag:
                summary_tag = h4_tag.find_next_sibling("p")
                summary = summary_tag.get_text(strip=True) if summary_tag else "No summary available"
            else:
                summary = "No summary available"
            
            articles.append((title, url, summary))
        
        return articles
    
    except Exception as e:
        print("Error extracting articles:", e)
        return []


def check_rag_relevance(query, sources_url, sources_documents, llm):
    """Check if at least one retrieved document is relevant to the query."""
    if not sources_documents and not sources_url:
        return False
    
    # Extract content from documents
    document_contents = []
    for doc in sources_documents:
        if hasattr(doc, 'page_content') and doc.page_content:
            document_contents.append(doc.page_content)
    
    # Simplify the prompt to be extremely direct
    check_relevance_prompt = f"""
    User Query: "{query}"
    
    Retrieved Documents: {document_contents}
    
    Do ANY of these documents contain information related to the query, even if they contradict or fact-check the claim in the query?
    
    IMPORTANT: A document is relevant if it discusses the same event, claim, or topic mentioned in the query, WHETHER IT SUPPORTS OR REFUTES THE CLAIM.
    
    Answer ONLY with "True" or "False".
    """
    
    # Get response
    response = llm.invoke([HumanMessage(content=check_relevance_prompt)])
    result = response.content.strip()
    
    # Exact matching for "True" or "False"
    return result == "True"



def fetch_source_content(url):
    """
    Fetch and extract meaningful content from a source URL.
    
    Args:
        url (str): The URL of the source.
    
    Returns:
        str: Extracted content (or snippet) for similarity calculation.
    """
    try:
        response = requests.get(url, timeout=5)  # Fetch the page
        if response.status_code != 200:
            return ""  # Return empty string if fetch fails

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract the main content (customize based on website structure)
        paragraphs = soup.find_all("p")  # Get all paragraph tags
        content = " ".join([p.get_text() for p in paragraphs[:5]])  # Get first 5 paragraphs

        return content.strip()
    
    except Exception as e:
        print(f"Error fetching content from {url}: {e}")
        return ""


def prioritize_sources(user_query: str, sources: list, response_text: str=None) -> list:
    """
    Reorder sources based on similarity with user query and response text.
    
    Args:
        user_query (str): The original user question
        response_text (str): The generated response content
        sources (list): List of source URLs to prioritize
        
    Returns:
        list: Reordered list of sources with priority based on relevance
    """
    if not user_query or not response_text or not sources:
        return sources  # Return as-is if missing data

    # Extract source IDs (assuming format ends in `-<number>`)
    def extract_id(url):
        try:
            return int(url.rstrip('/').split('-')[-1])
        except (ValueError, IndexError):
            return 0

    # Fetch source content snippets (assuming we have a way to extract them)
    source_texts = [fetch_source_content(url) for url in sources]  # Implement fetch_source_content function
    
    # Prepare text inputs for similarity calculation
    texts = [user_query, response_text] + source_texts  # First two are query and response

    # Compute TF-IDF similarity
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Compute similarity scores
    query_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[2:]).flatten()  # Query vs Sources
    response_similarities = cosine_similarity(tfidf_matrix[1:2], tfidf_matrix[2:]).flatten()  # Response vs Sources

    # Combine scores (weighted sum, adjust weights if needed)
    combined_scores = 0.6 * query_similarities + 0.4 * response_similarities

    # Sort sources based on combined similarity score
    sorted_sources = [x for _, x in sorted(zip(combined_scores, sources), key=lambda pair: pair[0], reverse=True)]

    return sorted_sources
