import os
import json
import torch
import fitz  # PyMuPDF
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH  = "/domaine/city/modelhub/Babelith-model-huggingface-Into/Qwen/Qwen2.5-VL-7B-Instruct/main"
PDF_PATH    = "/mnt/bareme-de-remboursement-cardif.pdf"
OUTPUT_JSON = "extracted.json"
DPI            = 200
MAX_NEW_TOKENS = 512

# ── Load model (once) ────────────────────────────────────────────────────────
print("Loading processor and model...")
processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
model.to("cuda")
model.eval()
print("Model loaded.")

# ── PDF → PIL Images (in-memory) ─────────────────────────────────────────────
def pdf_to_images(pdf_path, dpi=DPI):
    doc = fitz.open(pdf_path)
    images = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        max_side = max(img.width, img.height)
        img_square = Image.new("RGB", (max_side, max_side), "white")
        img_square.paste(img, (0, 0))
        images.append(img_square)
        print(f"  Loaded page {i+1} ({img_square.width}x{img_square.height})")
    return images

# ── Extract text from a single image ─────────────────────────────────────────
def extract_text(image):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": "Extract all visible text from this French document page. Return plain text only.",
                },
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[prompt],
        images=[image],
        padding=True,
        return_tensors="pt",
    )

    # BatchFeature has a .to() method that correctly moves ALL internal tensors
    inputs = inputs.to("cuda")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            use_cache=True,
        )

    generated_ids = output[:, inputs["input_ids"].shape[1]:]
    result = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    return result.strip()

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\nOpening PDF: {PDF_PATH}")
    images = pdf_to_images(PDF_PATH)
    print(f"Total pages: {len(images)}\n")

    all_text = []
    for i, img in enumerate(images):
        print(f"Processing page {i+1}/{len(images)}...")
        page_text = extract_text(img)
        all_text.append({"page": i + 1, "text": page_text})
        print(f"  ✓ Page {i+1} done. Preview: {page_text[:80]}...\n")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_text, f, indent=4, ensure_ascii=False)

    print(f"\nDone! Text saved to {OUTPUT_JSON}")
