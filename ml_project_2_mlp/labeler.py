import json
import logging
import os
import time

from openai import OpenAI
from tqdm import tqdm

from ml_project_2_mlp.data import WebsiteData

# Setup ranked logger
logging.getLogger("httpx").setLevel(logging.WARNING)


class WebsiteLabeler:
    def __init__(
        self,
        name: str,
        data: WebsiteData,
    ):
        self.name = name
        self.data = data

        # Get data directory
        self.data_dir = self.data.data_dir
        self.labels_dir = os.path.join(
            self.data_dir,
            "labels",
            self.name,
        )
        self.labels_path = os.path.join(self.labels_dir, f"{self.data.name}.json")
        os.makedirs(self.labels_dir, exist_ok=True)

    def get_labels(self) -> dict:
        """
        Gets the labels for the data.

        Returns:
            labels (list[dict]): List of labels for the data.
        """
        return self.labels

    def _load_labels(self) -> dict:
        """
        Loads the labels from the labels directory.
        """
        assert os.path.exists(
            self.labels_path
        ), f"Labels not found at {self.labels_path}"
        with open(self.labels_path) as f:
            return json.load(f)


class HumanLabeler(WebsiteLabeler):
    def __init__(self, name: str, data: WebsiteData):
        super().__init__(name, data)

        # Load labels
        self.labels = self._load_labels()


class GPTLabeler(WebsiteLabeler):
    def __init__(
        self,
        name: str,
        data: WebsiteData,
        fewshot: bool = False,
        features: list[str] | None = None,
        num_sentences: int = 10,
        num_tags: int = 10,
        num_keywords: int = 10,
        relabel: bool = True,
        model: str = "gpt-3.5-turbo",
        seed: int = 42,
    ):
        """
        Loads the labels for a dataset that were generated from a labeler using the OpenAI API.
        If the labels are not yet present, or the relabel flag is set to True, the labels are generated
        first.

        Args:
            name (str): Name of the labeler.
            data (WebsiteData): WebsiteData object containing the data to label.
            fewshot (bool): Whether to use few-shot learning. Defaults to False.
            features (list[str], optiona): List of features to include in few-shot prompt. Defaults to None.
            num_sentences (int): Number of sentences to use in the prompt. Defaults to 10.
            num_tags (int): Number of tags to use in the prompt. Defaults to 10.
            num_keywords (int): Number of keywords to use in the prompt. Defaults to 10.
            relabel (bool): Whether to relabel the data. Defaults to True.
            model (str): Which model to use. Defaults to "gpt-3.5-turbo".
            seed (int): Seed for the API. Defaults to 42.
        """
        super().__init__(name, data)

        # Save parameters
        self.fewshot = fewshot
        self.features = features
        self.num_sentences = num_sentences
        self.num_tags = num_tags
        self.num_keywords = num_keywords
        self.relabel = relabel
        self.model = model
        self.seed = seed

        # Load categories
        self.categories = self._load_categories()

        # Load the labels if they exist
        if not relabel and os.path.exists(self.labels_path):
            self.labels = self._load_labels()
            return

        # Initialise the OpenAI API
        api_key = os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)

        # Construct the prompt
        self.system_prompt = {
            "role": "system",
            "content": "You are an expert in website topic classification that accurately predicts the topic of a webpage based on features of the website, such as the TLD, domain, meta-tags, text, and more. Analyze the provided website data and classify it into relevant categories. Output a JSON string with categories as keys and binary values (0 or 1) indicating relevance.",
        }

        # Add few-shot example to prompt
        if self.fewshot:
            example_website = self._load_example_website()
            example_labels = self._load_example_labels()
            example_features = self._extract_features(example_website)

            self.system_prompt["content"] += (
                "Here is an example: Given website data: "
                + json.dumps(example_features)
                + " a good classification is: "
                + json.dumps(example_labels)
            )

        # Annotate websites
        websites = self.data.get_processed_data()
        self.labels = self._label_websites(websites)

        # Save the labels
        self._save_labels()

    def _load_labels(self) -> dict:
        """
        Loads the labels from the labels directory.
        """
        with open(self.labels_path) as f:
            return json.load(f)

    def _save_labels(self) -> None:
        assert self.labels is not None
        with open(self.labels_path, "w") as f:
            json.dump(self.labels, f)

    def _load_categories(self) -> list[str]:
        path = os.path.join(self.data_dir, "meta", "categories.txt")
        with open(path) as f:
            return [line.strip() for line in f]

    def _load_example_website(self) -> dict:
        """
        Loads the example website data.
        """
        path = os.path.join(self.data_dir, "meta", "example-website.json")
        with open(path) as f:
            return json.load(f)

    def _load_example_labels(self) -> dict:
        """
        Loads the labels for the example website data.
        """
        path = os.path.join(self.data_dir, "meta", "example-labels.json")
        with open(path) as f:
            return json.load(f)

    def _extract_features(self, website: dict) -> dict:
        """
        Extract the features from the website data that the labeler has access to
        and that are specified in the features list. Also, only include the specified
        number of sentences, tags and keywords.

        Args:
            website (dict): Dictionary with the example website data.

        Returns:
            website (dict): Dictionary with the example website data with only the specified features.
        """
        # Only include the specified number of sentences
        website["sentences"] = website["sentences"][: self.num_sentences]
        website["keywords"] = website["keywords"][: self.num_keywords]
        website["metatags"] = website["metatags"][: self.num_tags]

        # Only include the specified features
        example_website = {k: v for k, v in website.items() if k in self.features}

        return example_website

    def _label_websites(self, websites: list[dict]) -> list[dict]:
        """
        Annotates a list of websites using the labeler.

        Args:
            websites (list[dict]): List of websites to annotate.

        Returns:
            predictions (list[dict]): List of dictionaries with the classification results, namely the keys in each dict are:
                - input: The input website data.
                - output: The classification output.
                - is_valid: Whether the classification output is valid.
                - reason_invalid: Reason why the classification output is invalid.
                - wid: The website id.
        """
        labels = {}
        for wid, website in tqdm(websites.items()):
            # Label the website
            label = self._label_website(website)

            # Store the prediction
            labels[wid] = label
        return labels

    def _label_website(self, website: dict) -> dict:
        """
        Labels a single website using the labeler.

        Args:
            website (dict): Dictionary with the website data.

        Returns:
            output (dict): Dictionary with the labeling results.
        """
        # Construct the example
        website_features = self._extract_features(website)

        # Define the messages to send to the API
        messages = [
            self.system_prompt,
            {
                "role": "user",
                "content": json.dumps(website_features, ensure_ascii=False),
            },
        ]

        # Send the messages to the API
        start = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            seed=self.seed,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        duration = time.time() - start

        # Parse the response
        json_output = json.loads(response.choices[0].message.content)
        is_valid, reason_invalid = self._check_format(json_output)

        # If valid format, one hot encode the categories
        if is_valid:
            labels = [json_output[category] for category in self.categories]
        else:
            labels = [0] * len(self.categories)

        # Construct the output
        output = {
            "features": website_features,
            "labels": labels,
            "is_valid": is_valid,
            "reason_invalid": reason_invalid,
            "duration": duration,
        }

        return output

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
