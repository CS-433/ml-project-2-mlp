import json

from openai import OpenAI
from tqdm import tqdm


class GPTLabeler:
    # Categories for prediction
    categories = [
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

    def __init__(self, client: OpenAI, features: list[tuple[str, int]]):
        """
        Args:
            client (OpenAI): OpenAI client object.
            features (list[tuple[str, int]]): List of tuples of features and their downsampled count.
        """

        #  User defined attributes
        self.client = client
        self.features = features

        # System prompt
        self.example_website_data = {
            "title": "The New York Times - Breaking News, World News & Multimedia",
            "description": "Find breaking news, multimedia, reviews & opinion on Washington, business, sports, movies, travel, books, jobs, education, real estate, cars & more.",
            "keywords": [
                "breaking news",
                "world news",
                "politics",
                "economy",
                "sports",
                "arts",
                "movies",
                "travel",
                "books",
                "education",
                "real estate",
                "cars",
            ],
            "links": [
                "breaking-news",
                "world-news",
                "business",
                "sports",
                "arts",
                "travel",
            ],
            "tld": "com",
            "domain": "nytimes.com",
            "metatags": ["NYT", "news", "current events", "global news", "media"],
            "sentences": [
                "Breaking news: A major political development reshapes the landscape in Washington.",
                "Explore the latest world news and stay informed about global events.",
                "In-depth analysis of the current political and economic climate.",
                "Sports enthusiasts rejoice as the latest scores and highlights unfold.",
                "Discover the arts scene with reviews and insights into movies and cultural events.",
                "Plan your next adventure with our travel guides and recommendations.",
                "Insights into the business world, from market trends to corporate strategies.",
                "Education matters: Stay updated on developments in the field of academia.",
                "Real estate trends and property insights for homeowners and investors.",
                "Rev up your engines with the latest updates on cars and automotive industry.",
            ],
        }
        # Construct the example website data
        self._construt_example()

        self.example_output = {
            "Arts": 1,
            "Business": 1,
            "Computers": 0,
            "Games": 0,
            "Health": 1,
            "Home": 0,
            "Kids_and_Teens": 0,
            "News": 1,
            "Recreation": 0,
            "Reference": 0,
            "Science": 1,
            "Shopping": 0,
            "Society": 1,
            "Sports": 1,
        }

        self.system_prompt = {
            "role": "system",
            "content": "You are an assistant skilled in website classification using HTML content. \
                Analyze the provided website data and classify it into relevant categories. Output a JSON string with categories as \
                keys and binary values (0 or 1) indicating relevance. Here is an example: Given website data: "
            + json.dumps(self.example)
            + " a possible classification is: "
            + json.dumps(self.example_output),
        }

    def _construt_example(self):
        """
        Constructs the example website data.
        """
        self.example = {
            feature: self.example_website_data[feature]
            for feature, count in self.features
        }

    def _downsample_features(self, websites_feat: list[dict]) -> list[dict]:
        """
        Downsamples the features of the websites to the specified count.

        E.g. We have 100 links and the count is 10. We will only use the first 10 links.

        Args:
            websites_feat (list[dict]): List of website data dictionaries including websites to classify with their features.

        Returns:
            websites_feat_reduced (list[dict]): List of website data dictionaries including websites to classify with their features downsampled to the specified count.
        """

        # Go over the websites and downsample their features
        websites_feat_reduced = []
        for website_feat in websites_feat:
            # Go over the features and downsample them if count is specified
            for feature, count in self.features:
                if count is not None and website_feat[feature]:
                    max_count = min(count, len(website_feat[feature]))
                    website_feat[feature] = website_feat[feature][:max_count]

            # Save the downsampled website
            websites_feat_reduced.append(website_feat)

        return websites_feat_reduced

    def predict(self, websites_feat: list[dict]) -> list[dict]:
        """
        Args:
            websites_feat (list[dict]): List of website data dictionaries including websites to classify with their features.

        Returns:
            predictions (list[dict]): List of dictionaries with the classification results, namely the keys in each dict are:
                - input: The input website data.
                - output: The classification output.
                - is_valid: Whether the classification output is valid.
                - reason_invalid: Reason why the classification output is invalid.
                - wid: The website id.
        """

        # Downsample features - if count is None, do not downsample
        websites_feat_reduced = self._downsample_features(websites_feat)

        # Classify websites
        predictions = []
        try:
            for website in tqdm(websites_feat_reduced):
                # Get the features of the website based on the provided context features
                features = {feat: website[feat] for feat, _ in self.features}

                # Classify the website
                pred = self._classify_single_website(features)
                pred["wid"] = website["wid"]

                # Save the prediction
                predictions.append(pred)

        except Exception as e:
            print(e)
            return predictions

        return predictions

    def _check_format(self, output: dict) -> tuple[bool, str]:
        """
        Checks if the output is in the correct format.

        Args:
            output (dict): Dictionary with the classification results.

        Returns:
            is_valid (bool): Whether the output is in the correct format.
            reason_invalid (str): Reason why the output is not in the correct format.
        """

        # Returned valid categories
        all_valid = all(category in self.categories for category in output.keys())
        if not all_valid:
            return False, "Invalid categories"

        # Check if all values are binary
        all_binary = all(value in [0, 1] for value in output.values())
        if not all_binary:
            return False, "Non-binary values"

        return True, None

    def _classify_single_website(self, website_data: dict) -> dict:
        """
        Classifies a single website.

        Args:
            website_data (dict): Dictionary with the website data.

        Returns:
            output (dict): Dictionary with the classification results.
        """

        # Parse the input into json
        content = json.dumps(website_data, ensure_ascii=False)

        # Define the messages to send to the API
        messages = [
            self.system_prompt,
            {"role": "user", "content": content},
        ]

        # Send the messages to the API
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=messages,
            seed=42,  # For reproducibility
            max_tokens=200,
            response_format={"type": "json_object"},
        )

        # Parse the response
        json_output = json.loads(response.choices[0].message.content)
        is_valid, reason_invalid = self._check_format(json_output)

        # If valid format, one hot encode the categories
        if is_valid:
            preds = [json_output[category] for category in self.categories]
        else:
            preds = [0] * len(self.categories)

        # Construct the output
        output = {
            "input": website_data,
            "output": preds,
            "is_valid": is_valid,
            "reason_invalid": reason_invalid,
        }

        return output
