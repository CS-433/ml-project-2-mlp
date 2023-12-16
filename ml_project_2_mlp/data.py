import os
import pickle
from urllib.parse import urlparse

import pandas as pd
from bs4 import BeautifulSoup as Soup
from tld import get_tld
from tqdm import tqdm

import ml_project_2_mlp.utils as utils


class WebsiteData:
    """
    Dataset wrapper for scraped website data. Initialised with an
    identifier that is used to load a raw list of URLs from disk and
    then applies a series of transformations to the raw data to
    produce a processed dataset (scrape, parse, embed
    """

    def __init__(self, name: str, data_dir: str):
        # User defined attributes
        self.name = name
        self.data_dir = data_dir

        # Load raw, scraped, processed and embed data
        self.raw_data = self._load_raw_data()
        self.scraped_data = self._scrape_data()
        self.processed_data = self._process_data()
        self.embedded_data = self._embed_data()

    def get_raw_data(self):
        return self.raw_data

    def get_processed_data(self):
        return self.processed_data

    def get_embeddings(self):
        return self.embedded_data

    # ----------- Raw data
    def _load_raw_data(self):
        """
        Load the raw data from disk. The raw data is a CSV file
        containing a list of URLs to be scraped along with the corresponding
        website ID (wid).

        Returns:
            A Pandas DataFrame containing the raw data.
        """

        path = os.path.join(self.data_dir, "raw", f"{self.name}.csv")
        urls = pd.read_csv(path)
        return urls

    # ----------- Scraping
    def _scrape_data(self):
        """
        Scrape the raw website data and for each website save the following info:
        - wid: Unique identifier of the website
        - http_code: HTTP response code
        - is_valid: Whether the website is valid (i.e. HTTP response code is 200 and content type is text/html)
        - html: Raw HTML of the website
        - redirect_url: URL of the website after redirect (if any)
        - original_url: Original URL of the website (before redirect)
        """

        # Define the path where the scraped data will be/are saved
        save_dir = os.path.join(self.data_dir, "features", self.name)
        save_path = os.path.join(save_dir, "scraped.pkl")
        os.makedirs(save_dir, exist_ok=True)

        # Check if the scraped data already exists -> if not scrape, else load
        if not os.path.exists(save_path):
            websites = []
            for _, row in tqdm(
                list(self.raw_data.iterrows())[:5], total=len(self.raw_data[:5])
            ):
                # Get the website content
                result = self._get_website(row["url"], row["wid"])

                # Add to list
                websites.append(result)

            # Save the scraped data to disk
            with open(save_path, "wb") as f:
                pickle.dump(websites, f)

        else:
            with open(save_path, "rb") as f:
                websites = pickle.load(f)

        return websites

    @staticmethod
    def _is_valid_website(get_code, content_type):
        valid_get_code = get_code == 200
        valid_content_type = content_type.startswith("text/html")
        return valid_get_code and valid_content_type

    def _get_website(self, url, wid, timeout=10):
        """
        Get raw website content with additional meta info (e.g. response code, content type, etc.))
        for a given URL.

        Args:
            url: URL of the website
            wid: Unique identifier of the website
            timeout: Timeout in seconds

        Returns:
            A dictionary containing the raw website content and meta info.
        """

        # Get the website content
        response = utils.access_website(url, timeout=timeout)

        # Save the response to dict
        result = {
            "wid": wid,
            "http_code": None,
            "is_valid": False,
            "html": None,
            "redirect_url": None,
            "original_url": None,
        }

        # Add additional info to webpage object
        if response is not None:
            # Parse the response
            html, get_code, content_type, old_url, resp_url = response

            # Save the response code
            result["http_code"] = get_code

            # If valid, save remaining info
            if self._is_valid_website(get_code, content_type):
                result["is_valid"] = True
                result["html"] = html

                # Check for redirect
                final_domain = urlparse(resp_url).netloc
                original_domain = urlparse(old_url).netloc

                if final_domain != original_domain:
                    result["redirect_url"] = resp_url
                    result["original_url"] = old_url
                else:
                    result["redirect_url"] = None
                    result["original_url"] = old_url

        return result

    # ----------- Processing
    def _process_data(self) -> list[dict]:
        """
        Process the html and url raw data into useful features.

        Returns:
            web_features : list of dicts where each dict includes the needed info
        """
        # Store path
        save_dir = os.path.join(self.data_dir, "features", self.name)
        save_path = os.path.join(save_dir, "processed.pkl")
        os.makedirs(save_dir, exist_ok=True)

        # Check if the processed data already exists -> if not process, else load
        if not os.path.exists(save_path):
            # Filter webs to only include valid ones
            valid_webs = pd.DataFrame([w for w in self.scraped_data if w["is_valid"]])

            # Save the features
            web_features = []

            for i in tqdm(range(len(valid_webs))):
                # Get html
                html = valid_webs.iloc[i]["html"]

                # Get redirected url if available else original url
                url = (
                    valid_webs.iloc[i]["redirect_url"]
                    if valid_webs.iloc[i]["redirect_url"]
                    else valid_webs.iloc[i]["original_url"]
                )

                # Get id
                wid = {"wid": valid_webs.iloc[i]["wid"]}

                # Get features
                html_features = self._parse_html(html)
                url_features = self._parse_url(url)
                features = {**html_features, **url_features, **wid}

                # Save the features
                web_features.append(features)

                with open(save_path, "wb") as f:
                    pickle.dump(web_features, f)
        else:
            with open(save_path, "rb") as f:
                web_features = pickle.load(f)

        return web_features

    @staticmethod
    def _parse_html(html: str) -> dict:
        """
        Parse the HTML of a website and extract the following features:
        - title
        - description
        - keywords
        - links
        - sentences
        - metatags

        Args:
            html: HTML of the website

        Returns:
            A dictionary containing the extracted features
        """

        # Parse the HTML via BeautifulSoup
        soup = Soup(html, "html.parser")

        # Extract meta tags
        metatags = soup.findAll("meta")
        metatags = [m.get("name", None) for m in metatags]
        metatags = [m for m in metatags if m is not None]
        metatags = list(set([m.lower() for m in metatags]))

        # Extract site title
        title = soup.find("title")
        if title is not None:
            title = str(title.string)
            title = utils.clean_field(title)
            if len(title) == 0:
                title = None

        # Extract site description
        desc = soup.find("meta", attrs={"name": ["description", "Description"]})
        if not desc:
            desc = None
        else:
            content = desc.get("content", "")

            if len(content.strip()) == 0:
                desc = None
            else:
                desc = utils.clean_field(content)

        # Extract site keywords
        kw = soup.find("meta", attrs={"name": "keywords"})
        if not kw:
            kw = []
        else:
            kw = list(map(lambda s: s.strip(), kw.get("content", "").split()))
            if len(kw.strip()) == 0:
                kw = []

        # Extract site links
        a_tags = soup.find_all("a", href=True)
        links = [a.get("href", "") for a in a_tags]
        links = [utils.clean_link(link) for link in links]
        links = [link for link in links if len(link) != 0]
        links = [w.lower() for w in " ".join(links).split(" ") if len(w) != 0]
        if len(links) == 0:
            links = None

        # Extract text
        sentences = utils.split_in_sentences(soup)
        if len(sentences) == 0:
            sentences = None

        # Return the extracted features
        return {
            "title": title,
            "description": desc,
            "keywords": kw,
            "links": links,
            "sentences": sentences,
            "metatags": metatags,
        }

    @staticmethod
    def _parse_url(url: str) -> dict:
        """
        Parse the URL of a website and extract the following features:
        - tld
        - domain

        Args:
            url: URL of the website

        Returns:
            A dictionary containing the extracted features
        """

        # Get tld of url
        url_info = get_tld(url, as_object=True, fail_silently=True)

        # Get tld
        tld = url_info.tld

        # Get domain name
        domain = url_info.domain

        return {"tld": tld, "domain": domain}

    # ----------- Embedding
    def _embed_data(self):
        pass
