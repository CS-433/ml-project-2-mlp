"""
Module to access and load a webpage to be used by the homepage2vec model.

Includes:
    - TimeoutException: Exception to be raised when a timeout occurs.
    - time_limit: Context manager to set a time limit on the execution of a block.
    - access_website: Function to access a website and return its response.
"""

import signal
from contextlib import contextmanager

import requests


class TimeoutException(Exception):
    """Exception to be raised when a timeout occurs"""

    pass


@contextmanager
def time_limit(seconds):
    """Set a time limit on the execution of a block"""

    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def access_website(url, timeout=10):
    """
    Return the response corresponding to a url, or None if there was a request error
    """

    try:
        # avoid the script to be blocked
        with time_limit(10 * timeout):
            # change user-agent so that we don't look like a bot
            headers = requests.utils.default_headers()
            headers.update(
                {
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.16; rv:84.0) Gecko/20100101 Firefox/84.0",
                }
            )

            # r_head = requests.head("http://" + url, timeout=timeout, headers=headers)
            if not url.startswith("http://") and not url.startswith("https:"):
                url = "http://" + url
            r_get = requests.get(url, timeout=timeout, headers=headers)

            # head_code = r_head.status_code
            get_code = r_get.status_code
            if r_get.encoding.lower() != "utf-8":
                r_get.encoding = r_get.apparent_encoding
            text = r_get.text
            content_type = r_get.headers.get("content-type", "?").strip()
            return text, get_code, content_type

    except Exception as _:
        return None
