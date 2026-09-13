import json
from yolo_pipeline import run_image
from gemini_extractor import extract_fields
from pathlib import Path
import glob


def process_doc(img_path, output_path=None):
    file_name = Path(img_path).stem
    output_path  = f"output/{file_name}.json"

    image, confidence = run_image(img_path)
    data = extract_fields(image)
    data["confidence"] = round(confidence, 3) if confidence is not None else None
    data["needs_review"] = confidence is not None and confidence < 0.5


    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


    print(f"Saved extracted data to {output_path}")
    print(json.dumps(data, indent=2))
    return data

#if __name__ == "__main__":
    #process_doc("ADD SOURCE OF FILE")
