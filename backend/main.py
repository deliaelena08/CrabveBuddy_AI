import pickle
import os
import pandas as pd
import numpy as np
import torch
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
    global model_ro, model_en
    try:
        if os.path.exists(RO_MODEL_PATH):
            print(f"Loading fine-tuned RO model from {RO_MODEL_PATH}")
            model_ro = SentenceTransformer(RO_MODEL_PATH)
        else:
            print("Fine-tuned RO model not found. Using base.")
            model_ro = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

        print("AI Models loaded. Waiting for /sync-cakes...")
    except Exception as e:
        print(f"Error loading models: {e}")


@app.post("/sync-cakes")
async def sync_cakes(cakes: List[CakeDto]):
    global db_ro, db_en, embeddings_ro, embeddings_en

    if not cakes:
        return {"message": "Warning: Received empty list."}

    db_ro = cakes
    db_en = cakes

    texts_ro = []
    for c in cakes:
        text = (
            f"Prăjitură: {c.nameRo}. "
            f"Conține ingrediente: {c.ingredientsRo}. "
            f"Este {c.textureRo} și are aromă de {c.flavourRo}."
        )
        texts_ro.append(text)

    texts_en = []
    for c in cakes:
        text = (
            f"Dessert: {c.nameEn}. "
            f"Ingredients: {c.ingredientsEn}. "
            f"Texture: {c.textureEn}. Flavor: {c.flavourEn}."
        )
        texts_en.append(text)

    if model_ro:
        embeddings_ro = model_ro.encode(texts_ro, convert_to_tensor=True)

    if model_en:
        embeddings_en = model_en.encode(texts_en, convert_to_tensor=True)

    return {"message": f"Sync successful! Indexed {len(cakes)} cakes."}


@app.post("/recommend/ro")
async def recommend_ro(request: RecommendationRequest):
    if not db_ro or embeddings_ro is None:
        raise HTTPException(status_code=400, detail="Database empty.")

    return process_hybrid_search(db_ro, model_ro, embeddings_ro, request, "ro")


@app.post("/recommend/en")
async def recommend_en(request: RecommendationRequest):
    if not db_en or embeddings_en is None:
        raise HTTPException(status_code=400, detail="Database empty.")

    return process_hybrid_search(db_en, model_en, embeddings_en, request, "en")


def process_hybrid_search(db_source, model, embeddings, request, lang):
    query_vec = model.encode(request.query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_vec, embeddings)[0]
    scores = cos_scores.cpu().numpy()

    keywords = [w.lower() for w in request.query.split() if len(w) > 3]

    final_scores = []

    for idx, original_score in enumerate(scores):
        cake = db_source[idx]
        boost = 0.0

        if lang == "ro":
            cake_text = (f"{cake.nameRo} {cake.ingredientsRo} {cake.flavourRo}").lower()
        else:
            cake_text = (f"{cake.nameEn} {cake.ingredientsEn} {cake.flavourEn}").lower()

        for word in keywords:
            if word in cake_text:
                boost += 0.30
                if word in (cake.nameRo.lower() if lang == "ro" else cake.nameEn.lower()):
                    boost += 0.10

        final_scores.append(original_score + boost)

    all_indices = np.argsort(final_scores)[::-1]

    allergens_list = [a.strip().lower() for a in request.allergens.split(",") if a.strip()]
    recommendations = []

    for idx in all_indices:
        if len(recommendations) >= request.top_k:
            break

        cake = db_source[idx]

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
            "score": float(final_scores[idx]),
            "name": cake.nameRo if lang == "ro" else cake.nameEn,
            "ingredients": cake.ingredientsRo if lang == "ro" else cake.ingredientsEn,
            "fullData": cake
        })

    return {"results": recommendations}