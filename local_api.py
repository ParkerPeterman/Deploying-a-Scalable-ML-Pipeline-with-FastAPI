import json
import requests

# 1. GET Request
# The URL must match the root @app.get("/") in main.py
r_get = requests.get("http://127.0.0.1:8000")

# Print the status code and the extracted welcome message
print(f"Status Code: {r_get.status_code}")
# Extracting 'message' to match the output: "Hello from the API!"[cite: 6]
print(f"Result: {r_get.json()['message']}")

# 2. Data for Inference[cite: 6]
data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# 3. POST Request
# The URL must match the @app.post("/data/") defined in main.py[cite: 6]
r_post = requests.post("http://127.0.0.1:8000/data/", json=data)

# Print the status code and the prediction result[cite: 6]
print(f"Status Code: {r_post.status_code}")
# Extracting 'result' key as defined in main.py to get "<=50K"[cite: 6]
print(f"Result: {r_post.json()['result']}")