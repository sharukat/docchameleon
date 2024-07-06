# ==========================================================================================================================
# IMPORT DEPENDENCIES
# ==========================================================================================================================
import requests
from bs4 import BeautifulSoup


# ==========================================================================================================================
# MAIN CODE
# ==========================================================================================================================

def scraper(PackageName: str, TypeName: str):
    if TypeName == "JPanel":
        url = f"https://docs.oracle.com/javase/8/docs/api/javax/{PackageName}/{TypeName}.html"
    else:
        url = f"https://docs.oracle.com/javase/8/docs/api/java/{PackageName}/{TypeName}.html"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to retrieve the page: {url}")

    soup = BeautifulSoup(response.content, 'html.parser')
    result = soup.find('div', {'class': 'block'})
    return result

# Test
# result = scraper("util", "ArrayList")
# print(result)