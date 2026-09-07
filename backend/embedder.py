import io
from PIL import Image
from sentence_transformers import SentenceTransformer

class MultimodalEmbedder:
    def __init__(self):
        # Text encoder: multilingual-e5-large supports Thai very well (768 dims)
        print("Loading Thai/Multilingual Text Embedding model: multilingual-e5-large...")
        self.text_model = SentenceTransformer("intfloat/multilingual-e5-large")
        print("Text model loaded!")

        # Image encoder: CLIP ViT-B/32 for vision (512 dims)
        print("Loading Image Embedding model: clip-ViT-B-32...")
        self.image_model = SentenceTransformer("clip-ViT-B-32")
        print("Image model loaded!")

    def embed_text(self, text: str) -> list[float]:
        """Convert Thai/multilingual text into vector (768 dims)."""
        # multilingual-e5-large works best with "query: " prefix for queries
        embedding = self.text_model.encode(f"passage: {text}", normalize_embeddings=True)
        return embedding.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Convert user query into vector for search (768 dims)."""
        embedding = self.text_model.encode(f"query: {query}", normalize_embeddings=True)
        return embedding.tolist()

    def embed_image(self, image_bytes: bytes) -> list[float]:
        """Convert image bytes into vector using CLIP ViT (512 dims)."""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        embedding = self.image_model.encode(image)
        return embedding.tolist()

    def embed_image_query(self, query: str) -> list[float]:
        """Convert text into CLIP vector for cross-modal image search (512 dims)."""
        embedding = self.image_model.encode(query)
        return embedding.tolist()

# Lazy singleton
_embedder_instance = None

def get_embedder() -> MultimodalEmbedder:
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = MultimodalEmbedder()
    return _embedder_instance
