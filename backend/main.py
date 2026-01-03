import pickle
import os
import pandas as pd
import numpy as np
import torch  # Avem nevoie de torch pentru map_location
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util

app = FastAPI(title="CraveBuddy AI - Unified Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RO_MODEL_PATH = '../romanian_model/roberta_trained'
RO_PKL_EMB = '../romanian_model/recipe_embeddings_roberta.pkl'
RO_PKL_META = '../romanian_model/recipes_metadata.pkl'

# EN_MODEL_PATH = '../english_model/bert_trained'
EN_PKL_EMB = '../english_model/recipe_embeddings_en_bert.pkl'
EN_PKL_DATA = '../english_model/recipes_metadata_en.pkl'

model_ro = None
model_en = SentenceTransformer('all-MiniLM-L6-v2')
embeddings_ro = None
embeddings_en = None
db_ro = []
db_en = []


class CakeDto(BaseModel):
    id: int
    nameRo: str
    nameEn: Optional[str] = ""
    ingredientsRo: Optional[str] = ""
    ingredientsEn: Optional[str] = ""
    textureRo: Optional[str] = ""
    textureEn: Optional[str] = ""
    flavourRo: Optional[str] = ""
    flavourEn: Optional[str] = ""
    sweetness: Optional[int] = 0
    provenienceRo: Optional[str] = ""
    provenienceEn: Optional[str] = ""


class RecommendationRequest(BaseModel):
    query: str
    allergens: str = ""
    top_k: int = 3

@app.on_event("startup")
async def load_resources():
    global model_ro, model_en, embeddings_ro, embeddings_en, db_ro, db_en
    try:
        if os.path.exists(RO_MODEL_PATH):
            print(f"Loading fine-tuned RO model from {RO_MODEL_PATH}")
            model_ro = SentenceTransformer(RO_MODEL_PATH)
        else:
            print("Fine-tuned RO model not found, loading multilingual base...")
            model_ro = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')


        # if os.path.exists(EN_MODEL_PATH):
        #     print(f"Loading fine-tuned EN model from {EN_MODEL_PATH}")
        #     model_en = SentenceTransformer(EN_MODEL_PATH)

        if model_en is not None:
            print("Loaded English model: all-MiniLM-L6-v2")

        print("AI Models loaded. Waiting for /sync-cakes to receive database items...")
    except Exception as e:
        print(f"Error loading models: {e}")

@app.post("/sync-cakes")
async def sync_cakes(cakes: List[CakeDto]):
    global db_ro, db_en, embeddings_ro, embeddings_en, model_ro, model_en

    if not cakes:
        return {"message": "Warning: Received empty list."}

    db_ro = cakes
    db_en = cakes

    texts_ro = []
    for c in cakes:
        text = (
            f"Nume: {c.nameRo or ''}. "
            f"Ingrediente: {c.ingredientsRo or ''}. "
            f"Textură: {c.textureRo or ''}. "
            f"Aromă: {c.flavourRo or ''}."
        )
        texts_ro.append(text)

    texts_en = []
    for c in cakes:
        text = (
            f"{c.nameEn or ''}. "
            f"Ingredients: {c.ingredientsEn or ''}. "
            f"Characteristics: {c.textureEn or ''}, {c.flavourEn or ''}, {c.provenienceEn or ''}."
        )
        texts_en.append(text)

    if model_ro:
        embeddings_ro = model_ro.encode(texts_ro, convert_to_tensor=True)

    if model_en:
        embeddings_en = model_en.encode(texts_en, convert_to_tensor=True)

    return {"message": f"Sync successful! Updated both models with {len(cakes)} live items."}


@app.post("/recommend/ro")
async def recommend_ro(request: RecommendationRequest):
    global db_ro, embeddings_ro, model_ro

    if not db_ro or embeddings_ro is None:
        raise HTTPException(status_code=400, detail="RO Database empty. Run sync or check PKL files.")

    query_vec = model_ro.encode(request.query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_vec, embeddings_ro)[0]
    scores = cos_scores.cpu().numpy()

    return process_results(db_ro, scores, request.allergens, request.top_k, lang="ro")


@app.post("/recommend/en")
async def recommend_en(request: RecommendationRequest):
    global db_en, embeddings_en, model_en

    # Dacă db_en e gol, înseamnă că Java nu a trimis încă torturile
    if not db_en or embeddings_en is None:
        print("Atenție: Baza de date EN este goală. Java nu a făcut sync!")
        raise HTTPException(
            status_code=400,
            detail="AI-ul nu a primit încă datele din baza de date Java. Te rugăm să pornești serverul de backend Java."
        )

    query_vec = model_en.encode(request.query, convert_to_tensor=True)

    cos_scores = util.cos_sim(query_vec, embeddings_en)[0]
    scores = cos_scores.cpu().numpy()

    return process_results(db_en, scores, request.allergens, request.top_k, lang="en")


def process_results(db_source, scores, allergens_str, top_k, lang):
    allergens_list = [a.strip().lower() for a in allergens_str.split(",") if a.strip()]
    all_indices = np.argsort(scores)[::-1]

    recommendations = []

    for idx in all_indices:
        if len(recommendations) >= top_k:
            break

        if idx >= len(db_source):
            continue

        cake = db_source[idx]
        score = float(scores[idx])

        ingredients_check = (cake.ingredientsRo if lang == "ro" else cake.ingredientsEn) or ""
        ingredients_lower = ingredients_check.lower()

        has_allergen = False
        for allergen in allergens_list:
            if allergen in ingredients_lower:
                has_allergen = True
                break

        if has_allergen:
            continue

        recommendations.append({
            "id": cake.id,
            "score": score,
            "name": cake.nameRo if lang == "ro" else cake.nameEn,
            "ingredients": cake.ingredientsRo if lang == "ro" else cake.ingredientsEn,
            "fullData": cake
        })

    return {"results": recommendations}