import requests

endpoint = f"https://www.udemy.com/api-2.0/courses/2517920/?fields[course]=title"
response = requests.post(url=endpoint)
print(response)

