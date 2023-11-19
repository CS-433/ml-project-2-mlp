"""
import os

from homepage2vec.model import Webpage, WebsiteClassifier

from ml_project_2_mlp.conf import TESTS_PATH
from ml_project_2_mlp.homepage2vec.model import Webpage as Webpage2
from ml_project_2_mlp.homepage2vec.model import WebsiteClassifier as WebsiteClassifier2
"""

# Disabled because it takes too long to run
'''
def test_homepage2vec_output_equal():
    """
    Checks that the output of the local adaption of the Homepage2Vec
    model is the same as the original.
    """
    # Initialise both models
    model = WebsiteClassifier()
    model2 = WebsiteClassifier2()

    url = "https://dlab.epfl.ch"
    webpage = Webpage(url)
    webpage2 = Webpage2(url)
    html_path = os.path.join(TESTS_PATH, "data", "dlab.html")
    with open(html_path, "r") as f:
        content = f.read()
        webpage.html = content
        webpage2.html = content

    # Predict the scores and embeddings
    scores, embedding = model.predict(webpage)
    scores2, embedding2 = model2.predict(webpage2)

    # Check that the scores are the same
    assert scores == scores2, "Scores are different."
    assert embedding == embedding2, "Embeddings are different."
'''
