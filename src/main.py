import sys
import os

# Ensure src/ is on the path so verification_engine imports work
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from verification_engine import HybridVerificationEngine

app = FastAPI(title="Fake News Detection API")

# CORS — allows the React dev server (localhost:5173) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup — not on every request
print("Loading verification engine...")
engine = HybridVerificationEngine()
print("Engine ready.")


class VerifyRequest(BaseModel):
    title: str
    body: str = ""   # optional — frontend sends "" if user only pastes a headline


@app.post("/api/verify")
async def verify(req: VerifyRequest):
    if not req.title.strip():
        raise HTTPException(status_code=400, detail="Title cannot be empty.")
    try:
        result = engine.verify(title=req.title, body=req.body)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": engine.ml_model is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)