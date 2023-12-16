class WebsiteData:
    """
    Dataset wrapper for scraped website data. Initialised with an
    identifier that is used to load a raw list of URLs from disk and
    then applies a series of transformations to the raw data to
    produce a processed dataset (scrape, parse, embed
    """

    def __init__(self, name: str, data_dir: str):
        self.name = name
        self.data_dir = data_dir

        # Load raw, process and embed data
        self.raw_data = self._load_raw_data()
        self.processed_data = self._scrape_data()
        self.embedded_data = self._embed_data()

    def get_raw_data(self):
        return self.raw_data

    def get_processed_data(self):
        return self.processed_data

    def get_embeddings(self):
        return self.embedded_data

    def _load_raw_data(self):
        pass

    def _scrape_data(self):
        pass

    def _embed_data(self):
        pass
