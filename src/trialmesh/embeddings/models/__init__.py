# src/trialmesh/embeddings/models/__init__.py

from trialmesh.embeddings.models.e5 import E5LargeV2
from trialmesh.embeddings.models.bge import BGELargeV15
from trialmesh.embeddings.models.clinical_bert import SapBERT, BioClinicalBERT
from trialmesh.embeddings.models.blue_bert import BlueBERT

# Registry of all available models
MODEL_REGISTRY = {
    "e5-large-v2": E5LargeV2,
    "bge-large-v1.5": BGELargeV15,
    "sapbert": SapBERT,
    "bio-clinicalbert": BioClinicalBERT,
    "bluebert": BlueBERT,
}