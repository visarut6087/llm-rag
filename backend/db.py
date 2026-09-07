import os
import uuid
import base64
import hashlib
import chromadb
from chromadb.config import Settings

DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")


def source_identity(source_name: str) -> str:
    """Create a stable, non-sensitive ID for a source name."""
    source_name = source_name or "unknown"
    digest = hashlib.sha256(source_name.encode("utf-8")).hexdigest()[:16]
    return f"source-{digest}"

class VectorDBManager:
    def __init__(self):
        os.makedirs(DB_DIR, exist_ok=True)
        self.client = chromadb.PersistentClient(path=DB_DIR)
        
        # Collection for text chunks
        self.text_collection = self.client.get_or_create_collection(
            name="text_chunks",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Collection for image embeddings
        self.image_collection = self.client.get_or_create_collection(
            name="images",
            metadata={"hnsw:space": "cosine"}
        )

    def add_text(
        self,
        text: str,
        embedding: list[float],
        source_name: str = "user_upload",
        chunk_id: str = None,
    ) -> str:
        source_name = source_name or "user_upload"
        doc_id = str(uuid.uuid4())
        self.text_collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{
                "source": source_name,
                "source_id": source_identity(source_name),
                "source_name": source_name,
                "chunk_id": chunk_id or doc_id,
                "type": "text",
            }]
        )
        return doc_id

    def add_image(self, image_bytes: bytes, embedding: list[float], image_name: str) -> str:
        doc_id = str(uuid.uuid4())
        # Store base64 representation in metadata for easy display
        b64_img = base64.b64encode(image_bytes).decode("utf-8")
        self.image_collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[image_name],
            metadatas=[{
                "name": image_name,
                "source": image_name,
                "source_id": source_identity(image_name),
                "source_name": image_name,
                "chunk_id": image_name,
                "b64": b64_img,
                "type": "image"
            }]
        )
        return doc_id

    def search_all(self, query_embedding: list[float], image_query_embedding: list[float] = None, n_results: int = 3) -> dict:
        text_results = []
        image_results = []
        
        # Search text using multilingual embedding (768 dims)
        if self.text_collection.count() > 0:
            res_text = self.text_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, self.text_collection.count())
            )
            if res_text and res_text.get("documents"):
                ids = res_text.get("ids", [[]])[0]
                documents = res_text["documents"][0]
                metadatas = res_text.get("metadatas", [[]])[0]
                distances = res_text.get("distances", [[]])[0]
                for rank, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), start=1):
                    document_id = ids[rank - 1] if rank - 1 < len(ids) else None
                    meta = meta or {}
                    source_name = meta.get("source_name") or meta.get("source") or "unknown"
                    text_results.append({
                        "document_id": document_id,
                        "source": source_name,
                        "source_id": meta.get("source_id") or source_identity(source_name),
                        "source_name": source_name,
                        "chunk_id": meta.get("chunk_id") or document_id,
                        # ChromaDB returns cosine distance for this collection.
                        "retrieval_score": dist,
                        "rank": rank,
                        "text": doc,
                        "content": doc,
                        "metadata": meta,
                        # Keep the original field for backward compatibility.
                        "distance": dist
                    })

        # Search images using CLIP embedding (512 dims) if available
        if self.image_collection.count() > 0 and image_query_embedding is not None:
            res_img = self.image_collection.query(
                query_embeddings=[image_query_embedding],
                n_results=min(n_results, self.image_collection.count())
            )
            if res_img and res_img.get("metadatas"):
                ids = res_img.get("ids", [[]])[0]
                metadatas = res_img["metadatas"][0]
                distances = res_img.get("distances", [[]])[0]
                for rank, (meta, dist) in enumerate(zip(metadatas, distances), start=1):
                    document_id = ids[rank - 1] if rank - 1 < len(ids) else None
                    meta = meta or {}
                    image_name = meta.get("name")
                    source_name = meta.get("source_name") or meta.get("source") or image_name or "unknown"
                    image_results.append({
                        "document_id": document_id,
                        "source": source_name,
                        "source_id": meta.get("source_id") or source_identity(source_name),
                        "source_name": source_name,
                        "chunk_id": meta.get("chunk_id") or document_id,
                        # ChromaDB returns cosine distance for this collection.
                        "retrieval_score": dist,
                        "rank": rank,
                        "text": image_name,
                        "name": image_name,
                        "b64": meta.get("b64"),
                        "distance": dist
                    })

        return {
            "texts": text_results,
            "images": image_results
        }

    def get_stats(self):
        return {
            "text_count": self.text_collection.count(),
            "image_count": self.image_collection.count()
        }

_db_instance = None

def get_db() -> VectorDBManager:
    global _db_instance
    if _db_instance is None:
        _db_instance = VectorDBManager()
    return _db_instance
