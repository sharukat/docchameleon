import json
import requests

API_KEY = "BmopG%29d9Thccirg4e%29CjOw%28%28"


# ==========================================================================================================================
# MAIN CODE
# ==========================================================================================================================

class StackExchange:
    def __init__(self) -> None:
        self.base_url = "https://api.stackexchange.com/2.3/"

    def get_response(self, url) -> dict:
        full_url = self.base_url + url
        response = requests.get(full_url)
        response = json.loads(response.text)
        return response

    
    def search(self, api_name) -> dict:
        # URL to retrieve the similar questions based on the title
        search_url = f"search/advanced?key={API_KEY}&order=desc&sort=relevance&q={api_name}&tagged=java&site=stackoverflow&filter=!WZ(vx44tVjUC.7acIfBrX0zp8UnfGrFLCd17-o1"
        response = self.get_response(url=search_url)
        return response

