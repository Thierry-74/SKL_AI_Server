from pydantic import BaseModel, Field
from typing import List, Optional

# Requette : 
#           state = liste de 512 floats représentant l'état du jeu
#           legal_moves = liste parmi les 128 mouvements possibles de ceux qui sont légaux

class InferRequest(BaseModel):
    state: List[float] = Field(..., description="Vecteur d'état de taille fixe")
    legal_moves: Optional[List[int]] = Field(None, description="Indices des coups légaux (optionnel)")

# Reponse :
#           policy = liste des 128 probabilités de réussite pour chacun des moves possibles
#           value = probabilité de gain estimée depuis cet etat

class InferResponse(BaseModel):
    policy: List[float]
    value: float
    version: str = "v1"
