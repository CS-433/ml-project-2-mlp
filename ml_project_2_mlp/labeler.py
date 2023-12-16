import json
import os

from openai import OpenAI
from tqdm import tqdm

from .data import WebsiteData


class GPTLabeler:
    def __init__(
        self,
        name: str,
        data: WebsiteData,
        fewshot: bool = False,
        features: list[str] | None = None,
        num_sentences: int = 10,
        num_tags: int = 10,
        relabel: bool = True,
        model: str = "gpt-3.5-turbo",
        seed: int = 42,
    ):
        """
        Initialises a GPTLabeler object to label websites using a LLM. The class uses the OpenAI API to
        get multi-label predictions for the topic of websites. Many parameters can be set to configure the
        labeler, which influence the behaviour of the labeler, e.g. whether to include an label example,
        which features to use, which model to use, etc.

        Args:
            data (WebsiteData): WebsiteData object containing the data to label.
            fewshot (bool): Whether to use few-shot learning. Defaults to False.
            features (list[str], optiona): List of features to include in few-shot prompt. Defaults to None.
            num_sentences (int): Number of sentences to use in the prompt. Defaults to 10.
            num_tags (int): Number of tags to use in the prompt. Defaults to 10.
            relabel (bool): Whether to relabel the data. Defaults to True.
        """
        # Save parameters
        self.name = name
        self.data = data
        self.fewshot = fewshot
        self.features = features
        self.num_sentences = num_sentences
        self.num_tags = num_tags
        self.relabel = relabel
        self.seed = seed

        # Get data directory
        self.data_dir = self.data.data_dir
        self.labels_dir = os.path.join(
            self.data_dir,
            "labels",
            self.name,
        )
        self.labels_path = os.path.join(self.labels_dir, f"{self.data.name}.json")
        os.makedirs(self.labels_dir, exist_ok=True)

        # Load the labels if they exist
        if not relabel and not os.path.exists(self.labels_path):
            self.labels = self._load_labels()
            return

        # Initialise the OpenAI API
        api_key = os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)

        # Get the website data
        websites = self.data.get_processed_data()

        # Construct example
        example_website = self._load_example_website()
        print(example_website)
        example_labels = self._load_example_labels()
        print(example_labels)
        example_website = self._construct_example(example_website)

        # Construct the prompt
        self.system_prompt = {
            "role": "system",
            "content": "You are an expert in website topic classification that accurately predicts the topic of a webpage based on features of the website, such as the TLD, domain, meta-tags, text, and more. Analyze the provided website data and classify it into relevant categories. Output a JSON string with categories as keys and binary values (0 or 1) indicating relevance. Here is an example: Given website data: "
            + json.dumps(example_website)
            + " a good classification is: "
            + json.dumps(example_labels),
        }

        # Annotate
        self.labels = self._label_websites(websites)

        # Save the labels
        self._save_labels()

    def get_labels(self) -> dict:
        """
        Gets the labels for the data.

        Returns:
            labels (list[dict]): List of labels for the data.
        """
        return self.labels

    def _save_labels(self):
        assert self.labels is not None
        with open(self.labels_path, "w") as f:
            json.dump(self.labels, f)

    def _load_example_website(self):
        """
        Loads the example website data.
        """
        path = os.path.join(self.data_dir, "meta", "example-website.json")
        with open(path) as f:
            return json.load(f)

    def _load_example_labels(self):
        """
        Loads the annotation for the example website data.
        """
        path = os.path.join(self.data_dir, "meta", "example-labels.json")
        with open(path) as f:
            return json.load(f)

    def _construct_example(self, example_website: dict):
        """
        Only include the features that are specified in the features list
        for the few-shot prompt. Additionally, only include the specified number
        of sentences and tags.

        Args:
            example_website (dict): Dictionary with the example website data.
            features (list[str]): List of features to include in the few-shot prompt.
        """
        assert self.features is not None

        # Only include the specified features
        example_website = {
            k: v for k, v in example_website.items() if k in self.features
        }

        # Only include the specified number of sentences
        example_website["sentences"] = example_website["sentences"][
            : self.num_sentences
        ]
        example_website["tags"] = example_website["tags"][: self.num_tags]

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
        # Classify websites
        predictions = {}
        try:
            for website in tqdm(websites):
                # Downsample features - if count is None, do not downsample
                wid = website["wid"]
                website = self._construct_example(website)

                # Classify the website
                pred = self._label_website(website)

                # Save the prediction
                predictions[wid] = pred

        except Exception as e:
            print(e)
            return predictions

        return predictions

    def _label_website(self, website: dict) -> dict:
        """
        Labels a single website using the labeler.

        Args:
            website (dict): Dictionary with the website data.

        Returns:
            output (dict): Dictionary with the labeling results.
        """

        # Parse the input into json
        content = json.dumps(website, ensure_ascii=False)

        # Define the messages to send to the API
        messages = [
            self.system_prompt,
            {"role": "user", "content": content},
        ]

        # Send the messages to the API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            seed=self.seed,
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
            "input": website,
            "output": preds,
            "is_valid": is_valid,
            "reason_invalid": reason_invalid,
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
