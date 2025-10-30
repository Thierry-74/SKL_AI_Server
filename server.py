from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from schema import InferRequest, InferResponse
from model import InferenceEngine
from config import STATE_SIZE, NUM_MOVES

app = FastAPI(title="Klondike IA Server", version="v1")
engine = InferenceEngine()

# /health : pour vérifier l'état du serveur

@app.get("/health")
def health():
    return {"status": "ok", "state_size": STATE_SIZE, "num_moves": NUM_MOVES}

# /infer : pour lancer le moteur d'inférence
@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    try:
        policy, value = engine.infer(req.state, req.legal_moves) # lancement du moteur 
        return JSONResponse(content={
            "policy": policy.tolist(),
            "value": float(value),
            "version": "v1"
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"internal error: {e}")
