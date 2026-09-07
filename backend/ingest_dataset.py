import os
from embedder import get_embedder
from db import get_db

SUPPORTED_TEXT_EXTS = {".txt", ".md", ".json", ".csv"}
SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

def process_dataset_folder(dataset_dir: str = None) -> dict:
    """
    Scans a folder for text and image files and automatically ingests them into ChromaDB.
    """
    if dataset_dir is None:
        dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")

    os.makedirs(dataset_dir, exist_ok=True)
    
    embedder = get_embedder()
    db = get_db()
    
    processed_texts = 0
    processed_images = 0
    errors = []

    print(f"Scanning dataset folder: {dataset_dir}...")

    for root, _, files in os.walk(dataset_dir):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()

            try:
                # 1. Process Text Files
                if ext in SUPPORTED_TEXT_EXTS:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read().strip()
                    
                    if content:
                        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
                        for index, p in enumerate(paragraphs, start=1):
                            vec = embedder.embed_text(p)
                            db.add_text(
                                text=p,
                                embedding=vec,
                                source_name=file,
                                chunk_id=f"{file}#chunk-{index}",
                            )
                            processed_texts += 1
                        print(f"  [Text] Ingested: {file} ({len(paragraphs)} chunks)")

                # 2. Process Image Files (ViT)
                elif ext in SUPPORTED_IMAGE_EXTS:
                    with open(file_path, "rb") as f:
                        img_bytes = f.read()
                    
                    if img_bytes:
                        vec = embedder.embed_image(img_bytes)
                        db.add_image(image_bytes=img_bytes, embedding=vec, image_name=file)
                        processed_images += 1
                        print(f"  [Image] Ingested via ViT: {file}")

            except Exception as e:
                err_msg = f"Failed to ingest {file}: {str(e)}"
                print(f"  [Error] {err_msg}")
                errors.append(err_msg)

    return {
        "status": "success",
        "dataset_dir": dataset_dir,
        "processed_texts": processed_texts,
        "processed_images": processed_images,
        "errors": errors
    }

if __name__ == "__main__":
    result = process_dataset_folder()
    print("\n--- Ingestion Summary ---")
    print(f"Text Chunks Added : {result['processed_texts']}")
    print(f"Images Added (ViT): {result['processed_images']}")
    print(f"Errors            : {len(result['errors'])}")
