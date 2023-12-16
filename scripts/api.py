import json
from openai import OpenAI
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from sklearn.metrics import classification_report
from urllib.parse import urlparse


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


def parse_html(
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
    example_website_data = {
        "title": "Example Title",
        "meta_tags": {
            "keywords": "example, sample",
            "description": "An example website",
        },
        "content_preview": "Sample content from the website",
        "links": ["http://example.com/link1", "http://example.com/link2"],
        "additional_elements": "Example additional HTML elements",
    }

    example_output = {
        "Arts": 0,
        "Business": 0,
        "Computers": 1,
        "Games": 0,
        "Health": 0,
        "Home": 0,
        "Kids_and_Teens": 0,
        "News": 0,
        "Recreation": 0,
        "Reference": 1,
        "Science": 0,
        "Shopping": 0,
        "Society": 0,
        "Sports": 0,
    }

    messages = [
        {
            "role": "system",
            "content": "You are an assistant skilled in website classification using HTML content. Analyze the provided website data and classify it into relevant categories. Output a JSON string with categories as keys and binary values (0 or 1) indicating relevance. Here is an example: Given website data: "
            + json.dumps(example_website_data)
            + " a possible classification is: "
            + json.dumps(example_output),
        },
        {"role": "user", "content": json.dumps(website_data, ensure_ascii=False)},
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        seed=42,
        max_tokens=200,
        response_format={"type": "json_object"},
    )
    json_output = json.loads(response.choices[0].message.content)
    return json_output


def check_classification_format_json(classification_dict):
    """
    Checks if all categories in the classification result are valid and have binary values in JSON mode.
    """
    return all(
        category in CATEGORIES and classification_dict.get(category) in [0, 1]
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


def classify_and_validate(dataset, num_samples):
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
            scraped_data = parse_html(
                url,
                max_links=10,
                max_sentences=20,
                meta_tags=["description", "keywords"],
            )
        except Exception as e:
            print(f"Error scraping URL: {url}, error: {e}")
            scraped_data = None

        if scraped_data:
            classification_dict = classify_website(scraped_data)

            if check_classification_format_json(classification_dict):
                predictions.append(
                    [classification_dict[category] for category in CATEGORIES]
                )
                ids.append(index)
            else:
                print(
                    f"Format error in classification result for URL: {url}, result: {classification_dict}"
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


def check_website(url, min_content_length=100):
    try:
        url = convert_to_full_url(url)
        response = requests.get(url, timeout=5)

        # Check for redirect
        final_domain = urlparse(response.url).netloc
        original_domain = urlparse(url).netloc
        if final_domain != original_domain:
            return False, f"{url} redirected from the website"

        # Check for 404 Not Found
        if response.status_code == 404:
            return False, f"{url} is a 404"

        # Parse content
        soup = BeautifulSoup(response.content, "html.parser")
        text_content = soup.get_text().strip()

        # Check content length
        if len(text_content) < min_content_length:
            return (
                False,
                f"{url} has less content than the minimum required ({min_content_length} characters)",
            )

        return True, f"Website looks good"
    except requests.RequestException as e:
        return False, f"{url} raised an exception: {e}"


def scrape_websites_from_dataframe(
    df,
    url_column,
    max_links=None,
    max_sentences=None,
    user_agent=None,
    timeout=10,
    meta_tags=None,
    html_elements=None,
):
    """
    Iterates through a DataFrame and scrapes content from URLs in a specified column.

    Parameters:
    df (DataFrame): The DataFrame containing URLs.
    url_column (str): The column name in the DataFrame where URLs are stored.
    max_links, max_sentences, user_agent, timeout, meta_tags, html_elements: Parameters to pass to scrape_website.

    Returns:
    DataFrame: The original DataFrame with additional columns for scraped content.
    """
    # Ensure the URL column exists in the DataFrame
    if url_column not in df.columns:
        raise ValueError(f"Column '{url_column}' not found in DataFrame")

    # Initialize columns for scraped data
    df["Title"] = None
    df["Links"] = None
    df["Meta Tags"] = None
    df["Content Preview"] = None
    df["Additional Elements"] = None

    # Iterate over the DataFrame
    for index, row in df.iterrows():
        url = row[url_column]
        try:
            scraped_data = parse_html(
                url,
                max_links,
                max_sentences,
                user_agent,
                timeout,
                meta_tags,
                html_elements,
            )
            df.at[index, "Title"] = scraped_data["title"]
            df.at[index, "Links"] = scraped_data["links"]
            df.at[index, "Meta Tags"] = scraped_data["meta_tags"]
            df.at[index, "Content Preview"] = scraped_data["content_preview"]
            df.at[index, "Additional Elements"] = scraped_data["additional_elements"]
        except Exception as e:
            print(f"Error scraping {url}: {e}")

    return df


def fetch_url_data(df, url_column):
    # Add new columns to the DataFrame
    df["Response Status"] = None
    df["Redirected URL"] = None
    df["Response Text"] = None

    # Iterate through each row
    for index, row in df.iterrows():
        try:
            url = row[url_column]
            url = convert_to_full_url(url)
            response = requests.get(url, allow_redirects=True)

            # Update the DataFrame with the response details
            df.at[index, "Response Status"] = response.status_code
            df.at[index, "Redirected URL"] = response.url
            df.at[index, "Response Text"] = response.text
        except Exception as e:
            print(f"Error fetching data for URL {url}: {e}")

    return df


def classify(dataset, num_samples):
    predictions = []
    ids = []
    failed_urls = []

    for index, row in dataset.head(num_samples).iterrows():
        url = convert_to_full_url(row["Input.url"])
        try:
            scraped_data = parse_html(
                row,
                "Response Text",
                max_links=10,
                max_sentences=20,
                meta_tags=["description", "keywords"],
            )
            print(scraped_data)
        except Exception as e:
            print(f"Error scraping URL: {url}, error: {e}")
            scraped_data = None

        if scraped_data:
            classification_dict = classify_website(scraped_data)

            if check_classification_format_json(classification_dict):
                predictions.append(
                    [classification_dict[category] for category in CATEGORIES]
                )
                ids.append(index)
            else:
                print(
                    f"Format error in classification result for URL: {url}, result: {classification_dict}"
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


def parse_html(
    row,
    column,
    max_links=None,
    max_sentences=None,
    meta_tags=None,
    html_elements=None,
):
    response = row[column]
    if response is None:
        return None
    soup = BeautifulSoup(response, "html.parser")

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
        "url": row["Input.url"],
        "title": soup.title.string if soup.title else None,
        "links": list(links.items())[:max_links],
        "meta_tags": meta_tags,
        "content_preview": " ".join(sentences[:max_sentences])
        if max_sentences
        else " ".join(sentences),
        "additional_elements": elements,
    }
