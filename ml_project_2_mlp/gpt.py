import json
from typing import Dict, List, Tuple
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import pandas as pd
from openai import OpenAI
from sklearn.metrics import classification_report


class GPT:
    def __init__(
        self, categories: List[str], client: OpenAI, features: List[Tuple[str, int]]
    ):
        self.categories = categories
        self.client = client
        self.predictions = None
        self.features = features

    def _downsample_features(self, X: list[Dict]):
        # is a dict of dicts that have the feature keys
        X_new = []
        for website in X:
            new_value = {}
            new_value["id"] = website["id"]
            for feature, count in self.features:
                if count is not None:
                    new_value[feature] = website[feature][:count]
                else:
                    new_value[feature] = website[feature]
            X_new.append(new_value)
        return X_new

    def predict(self, X: list[Dict]):
        X = self._downsample_features(X)
        if not self.predictions:
            self.predictions = []

        for website in X:
            features = [website[feature] for feature, _ in self.features]
            prediction, error = self._classify_single_website(features)
            self.predictions.append(
                {"id": website["id"], "prediction": prediction, "error": error}
            )
        return self.predictions

    def validate(self, Y_actual):
        if self.predictions is None:
            raise ValueError("No predictions found. Run predict method first.")
        # get df from a list of dicts
        predictions_df = pd.DataFrame(self.predictions)
        predictions_df = predictions_df[predictions_df["error"].isna()]
        categories_df = pd.DataFrame(
            predictions_df["prediction"].tolist(),
            index=predictions_df.id,
            columns=self.categories,
        )
        Y_actual = Y_actual.loc[predictions_df.index]
        report = classification_report(Y_actual[self.categories], categories_df)
        return report

    def _classify_single_website(self, website_data):
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
        print(type(website_data))
        print(website_data)
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=messages,
            seed=42,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        json_output = json.loads(response.choices[0].message.content)
        valid_format = self._check_format(json_output)
        if valid_format:
            return [json_output[category] for category in self.categories], None
        else:
            return None, "Invalid format"

    def _check_format(self, classification_dict):
        """
        Checks if all categories in the classification result are valid and have binary values in JSON mode.
        """
        return all(
            category in self.categories and classification_dict.get(category) in [0, 1]
            for category in self.categories
        )
