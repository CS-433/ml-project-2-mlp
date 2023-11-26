from openai import OpenAI
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from sklearn.metrics import classification_report

client = OpenAI()

CATEGORIES = [
    "Arts",
    "Business",
    "Computers",
    "Games",
    "Health",
    "Home",
    "Kids_and_Teens",
    "News",
    "Recreation",
    "Reference",
    "Science",
    "Shopping",
    "Society",
    "Sports",
]


def scrape_website(
    url,
    max_links=None,
    max_sentences=None,
    user_agent=None,
    timeout=10,
    meta_tags=None,
    html_elements=None,
):
    """
    Scrapes content from a website URL.

    Parameters:
    url (str): The URL of the website to scrape.
    max_links (int, optional): The maximum number of links to extract.
    max_sentences (int, optional): The maximum number of sentences to extract from the content.
    user_agent (str, optional): The user agent string to use for the request.
    timeout (int): The timeout for the request in seconds.
    meta_tags (list, optional): A list of meta tag names to extract.
    html_elements (list, optional): A list of HTML elements to extract.

    Returns:
    dict: A dictionary containing the scraped content.
    """
    headers = {"User-Agent": user_agent} if user_agent else None
    response = requests.get(url, headers=headers, timeout=timeout)
    soup = BeautifulSoup(response.text, "html.parser")

    links = {a.text: a["href"] for a in soup.find_all("a", href=True)}
    meta_tags = {
        meta["name"]: meta["content"]
        for meta in soup.find_all("meta")
        if meta.get("name") and (not meta_tags or meta["name"] in meta_tags)
    }
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(soup.stripped_strings))
    elements = (
        {element: soup.find_all(element) for element in html_elements}
        if html_elements
        else {}
    )

    return {
        "url": url,
        "title": soup.title.string if soup.title else None,
        "links": list(links.items())[:max_links],
        "meta_tags": meta_tags,
        "content_preview": " ".join(sentences[:max_sentences])
        if max_sentences
        else " ".join(sentences),
        "additional_elements": elements,
    }


def classify_website(website_data):
    """
    Classifies a website into various categories based on its HTML content.

    Parameters:
    website_data (dict): A dictionary containing key data about the website, such as title, meta tags,
                         content preview, links, and other HTML elements.

    Returns:
    dict: A dictionary where each key is a category and its value is either 0 (Not Relevant) or 1 (Relevant).
          The keys are strings exactly matching the category names, with underscores for spaces where necessary.
    """
    messages = [
        {
            "role": "system",
            "content": "You are an assistant skilled in website classification using HTML content. Analyze the provided website data and classify it into relevant categories. Use the exact category names with underscores for spaces where necessary, and determine the website's relevance to each category as either 0 (Not Relevant) or 1 (Relevant).",
        },
        {
            "role": "user",
            "content": f"""
            Title: {website_data.get('title', 'Not Provided')}
            Meta Keywords: {website_data.get('meta_tags', {}).get('keywords', 'Not Provided')}
            Meta Description: {website_data.get('meta_tags', {}).get('description', 'Not Provided')}
            Content Preview: {website_data.get('content_preview', 'Not Provided')}
            Links: {website_data.get('links', 'Not Provided')}
            Additional Elements: {website_data.get('additional_elements', 'Not Provided')}

            Categories: Arts, Business, Computers, Games, Health, Home, Kids_and_Teens, News, Recreation, Reference, Science, Shopping, Society, Sports.

            Please classify the website into these categories, providing a binary classification (0 or 1) for each.
            """,
        },
    ]
    messages_nice = "\n".join([f"    {message['content']}" for message in messages])
    print(messages_nice)
    print(f"Total number of characters in messages: {len(messages_nice)}")
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages, seed=42, max_tokens=100
    )
    return completion.choices[0].message.content


def parse_classification_result(classification_str):
    """
    Parses the classification result string into a dictionary.

    Parameters:
    classification_str (str): The classification result string.
    categories (list): A list of categories to check in the classification result.

    Returns:
    dict: A dictionary with categories as keys and binary classification (0 or 1) as values.
    """
    lines = classification_str.strip().split("\n")
    classification_dict = {}

    for line in lines:
        if ":" not in line:
            print(f"Unexpected format in line: '{line}'")  # Logging the unexpected line
            break
        category, value = line.split(":")
        category, value = category.strip(), value.strip()

        if category in CATEGORIES and value in ["0", "1"]:
            classification_dict[category] = int(value)
        else:
            print(f"Invalid format in classification result: '{line}'")

    return classification_dict


def check_classification_format(classification_dict):
    """
    Checks if all categories in the classification result are valid and have binary values.

    Parameters:
    classification_dict (dict): The classification dictionary.
    categories (list): A list of valid categories.

    Returns:
    bool: True if the format is correct, False otherwise.
    """
    return all(
        category in classification_dict and classification_dict[category] in [0, 1]
        for category in CATEGORIES
    )


def convert_to_full_url(url):
    """
    Converts a URL to a full URL format if it doesn't start with http:// or https://.

    Parameters:
    url (str): The URL to be converted.

    Returns:
    str: The converted full URL.
    """
    return (
        url
        if url.startswith("http://") or url.startswith("https://")
        else "http://" + url
    )


def calssify_and_validate(dataset, num_samples):
    """
    Processes a dataset to classify websites and generate predictions.

    Parameters:
    dataset (DataFrame): The dataset containing website URLs and other data.
    num_samples (int): The number of samples to process from the dataset.

    Returns:
    tuple: A tuple containing the DataFrame with predictions and a list of failed URLs.
    """
    predictions = []
    ids = []
    failed_urls = []

    for index, row in dataset.head(num_samples).iterrows():
        url = convert_to_full_url(row["Input.url"])
        try:
            scraped_data = scrape_website(
                url,
                max_links=10,
                max_sentences=20,
                meta_tags=["description", "keywords"],
            )
        except Exception as e:
            print(f"Error scraping URL: {url}, error: {e}")
            scraped_data = None

        if scraped_data:
            classification_str = classify_website(scraped_data)
            result = parse_classification_result(classification_str)

            if check_classification_format(result):
                predictions.append([result[category] for category in CATEGORIES])
                ids.append(index)
            else:
                print(
                    f"Format error in classification result for URL: {url}, result: {result}"
                )
                failed_urls.append(url)
        else:
            failed_urls.append(url)

    predictions_df = pd.DataFrame(predictions, columns=CATEGORIES)
    predictions_df["Index"] = ids
    successful_df = dataset.loc[ids]
    report = classification_report(
        successful_df[CATEGORIES], predictions_df[CATEGORIES], target_names=CATEGORIES
    )
    return predictions_df, failed_urls, report
