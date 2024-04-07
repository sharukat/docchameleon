import os
import requests
import lib.config as config
from lib.common import stub
import modal

def split(list_a, chunk_size):
    """
    The split function takes a list and splits it into chunks of the specified size.
    Args:
        list_a: Specify the list that will be split into chunks
        chunk_size: Specify the size of each chunk
    Returns:
        A generator object
    """
    for i in range(0, len(list_a), chunk_size):
        yield list_a[i:i + chunk_size]


def check_url(url):
    """Checks if a URL is working by sending a HEAD request and verifying a 200 status code.
    Args:
        url: The URL to check.
    Returns:
        True if the URL is working, False otherwise.
    """
    try:
        response = requests.head(url)
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"Error checking URL: {e}")
        return False


def remove_broken_urls(url_list):
  """
  Iterates through a list of URLs, checks their status, and removes non-working ones.
  Args:
      url_list: A list containing URLs to be checked.
  Returns:
      A new list with only working URLs.
  """
  working_urls = []
  for url in url_list:
    if check_url(url):
      working_urls.append(url)
  return working_urls


def is_valid_api(text):
  """Checks if the input string starts with 'tf.' and raises a ValueError if not.
  Args:
    text: The string to check.
  Raises:
    ValueError: If the string does not start with 'tf.'.
  """
  if not text.startswith("tf."):
    raise ValueError(f"{config.COLOR['RED']}❗ EXPECTED API TO START WITH 'tf.', GOT '{text}' INSTEAD.{config.COLOR['ENDC']}\n")
  print(f"{config.COLOR['GREEN']}✅ {text} API NAME IS VALID.{config.COLOR['ENDC']}\n")
  

# @stub.function(volumes={"/my_vol": modal.Volume.from_name("my-docs-volume", create_if_missing=True)})
def get_documentation(filename):
  """Reads a Markdown file and returns its contents as a string.
  Args:
    str: Name of the file without the extension
  Raises:
    ValueError: The documentation is not found.
  """
  filename_with_ext = os.path.join("/root/docs", filename) + ".md"  # Add the .md extension
  try:
    with open(filename_with_ext, "r", encoding="utf-8") as file:
      content = file.read()
    print(f"{config.COLOR['GREEN']}🔍 FOUND {filename} API DOCUMENTATION.{config.COLOR['ENDC']}\n")
    return content
  except FileNotFoundError:
    raise ValueError(f"{config.COLOR['RED']} ❗ DOCUMENTATION NOT FOUND.{config.COLOR['ENDC']}\n")