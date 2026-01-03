import requests
import json

JAVA_API_URL = "http://localhost:8080/api/cakes"
PYTHON_API_URL = "http://127.0.0.1:8000/sync-cakes"

def run_sync():
    try:
        print(f"Downloading data from Java ({JAVA_API_URL})...")
        response = requests.get(JAVA_API_URL)

        if response.status_code != 200:
            print("Error connecting to Java:", response.status_code)
            return

        cakes = response.json()
        print(f"Found {len(cakes)} cakes in database.")

        print(f"Sending data to Python AI ({PYTHON_API_URL})...")
        response_ai = requests.post(PYTHON_API_URL, json=cakes)

        if response_ai.status_code == 200:
            print("SUCCESS:", response_ai.json()['message'])
        else:
            print("Error at Python:", response_ai.text)

    except Exception as e:
        print("Connection error:", e)
        print("Ensure both servers (Java and Python) are running!")

if __name__ == "__main__":
    run_sync()