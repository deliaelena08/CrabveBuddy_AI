from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util
import numpy as np

app = FastAPI(title="CraveBuddy AI - Full Integration")

# Configurare CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 1. MEMORIA SERVERULUI
# ==========================================
db_cakes = []
db_embeddings = None
model = None


@app.on_event("startup")
async def load_model():
    global model
    print("Se încarcă modelul AI (Multilingual)...")
    # Folosim modelul multilingual pentru a înțelege și româna și engleza
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("Model încărcat!")


# ==========================================
# 2. DEFINIREA DATELOR (Sincronizat cu Java CakeDto)
# ==========================================
# Trebuie să aibă EXACT aceleași nume de câmpuri ca JSON-ul trimis de Java
class CakeDto(BaseModel):
    id: int
    name: str
    # Folosim Optional[...] = "" ca să acceptăm NULL din Java
    ingredients: Optional[str] = ""
    texture: Optional[str] = ""
    flavour: Optional[str] = ""
    sweetness: Optional[int] = 0
    provenience: Optional[str] = ""


class RecommendationRequest(BaseModel):
    query: str
    top_k: int = 3


# ==========================================
# 3. ENDPOINT DE SINCRONIZARE (Updated)
# ==========================================
@app.post("/sync-cakes")
async def sync_cakes(cakes: List[CakeDto]):
    global db_cakes, db_embeddings, model

    if not cakes:
        return {"message": "Lista primită este goală."}

    print(f"📥 Primit {len(cakes)} torturi din Java. Se calculează vectorii...")

    db_cakes = cakes

    # --- FEATURE ENGINEERING ---
    texts_to_embed = []
    for cake in cakes:
        # Folosim "or ''" pentru a transforma None în string gol,
        # altfel ar scrie "None" în textul analizat de AI
        i_safe = cake.ingredients or ""
        t_safe = cake.texture or ""
        f_safe = cake.flavour or ""
        s_safe = cake.sweetness or 0
        p_safe = cake.provenience or ""

        combined_text = (
            f"Nume: {cake.name}. "
            f"Ingrediente: {i_safe}. "
            f"Textura: {t_safe}. "
            f"Aroma: {f_safe}. "
            f"Dulceata: {s_safe}. "
            f"Provenienta: {p_safe}."
        )
        texts_to_embed.append(combined_text)

    # Generăm Embeddings
    if model:
        db_embeddings = model.encode(texts_to_embed, convert_to_tensor=True)

    return {
        "message": f"✅ Sincronizare reușită! AI-ul a învățat {len(cakes)} torturi (chiar și pe cele cu date lipsă)."}

# ==========================================
# 4. ENDPOINT DE RECOMANDARE (React -> Python)
# ==========================================
# 1. Updatezi modelul de Request
class RecommendationRequest(BaseModel):
    query: str
    allergens: str = ""  # Primim alergenii ca un string (ex: "nuci, gluten")
    top_k: int = 3


# 2. Modifici Endpoint-ul
@app.post("/recommend")
async def recommend(request: RecommendationRequest):
    global db_cakes, db_embeddings, model

    if not db_cakes or db_embeddings is None:
        raise HTTPException(status_code=400, detail="AI-ul este gol. Execută Sync mai întâi!")

    # A. Vectorizăm cererea utilizatorului
    query_vec = model.encode(request.query, convert_to_tensor=True)

    # B. Calculăm similaritatea
    cos_scores = util.cos_sim(query_vec, db_embeddings)[0]
    scores = cos_scores.cpu().numpy()

    # C. LOGICA DE FILTRARE (HIBRIDĂ)
    # Lista de alergeni de exclus (curățăm textul primit)
    allergens_list = [a.strip().lower() for a in request.allergens.split(",") if a.strip()]

    # D. Sortăm descrescător după scor
    # Obținem toți indicii, nu doar top K, pentru a putea filtra
    all_indices = np.argsort(scores)[::-1]

    recommendations = []

    for idx in all_indices:
        # Dacă am strâns deja destule recomandări, ne oprim
        if len(recommendations) >= request.top_k:
            break

        cake = db_cakes[idx]
        score = float(scores[idx])

        # --- VERIFICARE ALERGENI ---
        # Dacă userul a zis "nuci", verificăm dacă "nuci" apare în ingredientele prăjiturii
        has_allergen = False
        if allergens_list:
            ingredients_lower = (cake.ingredients or "").lower()
            for allergen in allergens_list:
                # Verificăm simplist dacă cuvântul apare (ex: "nuci")
                if allergen in ingredients_lower:
                    has_allergen = True
                    break

        # Dacă are alergeni, SĂRIM peste această prăjitură (nu o recomandăm)
        if has_allergen:
            continue

        recommendations.append({
            "id": cake.id,
            "title": cake.name,
            "ingredients": cake.ingredients,
            "category": cake.flavour,
            "score": score
        })

    return {"results": recommendations}