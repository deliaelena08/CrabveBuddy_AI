import requests
import json

# Configurare URL-uri
JAVA_API_URL = "http://localhost:8080/api/cakes"  # Verifică portul tău de Java!
PYTHON_API_URL = "http://127.0.0.1:8000/sync-cakes"


def run_sync():
    try:
        # 1. Luăm datele din Java
        print(f"🔄 Se descarcă datele din Java ({JAVA_API_URL})...")
        response = requests.get(JAVA_API_URL)

        if response.status_code != 200:
            print("❌ Eroare la conectarea cu Java:", response.status_code)
            return

        cakes = response.json()
        print(f"✅ Găsit {len(cakes)} torturi în baza de date.")

        # 2. Le trimitem la Python
        print(f"🔄 Se trimit datele la Python AI ({PYTHON_API_URL})...")
        response_ai = requests.post(PYTHON_API_URL, json=cakes)

        if response_ai.status_code == 200:
            print("🎉 SUCCES:", response_ai.json()['message'])
        else:
            print("❌ Eroare la Python:", response_ai.text)

    except Exception as e:
        print("❌ A apărut o eroare de conexiune:", e)
        print("Asigură-te că ambele servere (Java și Python) sunt pornite!")


if __name__ == "__main__":
    run_sync()