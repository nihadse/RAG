import os
import json
import torch
import fitz  # PyMuPDF
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH = "/domaine/city/modelhub/Babelith-model-huggingface-Into/Qwen/Qwen2.5-VL-7B-Instruct/main"
PDF_PATH   = "/mnt/bareme-de-remboursement-cardif.pdf"
OUTPUT_JSON = "extracted.json"

DPI           = 200        # ↓ from 400 — still sharp enough for text OCR
MAX_NEW_TOKENS = 512       # ↓ from 2048 — plenty for one page of text
BATCH_SIZE    = 1          # increase to 2-4 if you have enough VRAM

# ── Load model (once) ────────────────────────────────────────────────────────
print("Loading processor and model...")
processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    trust_remote_code=True,
    low_cpu_mem_usage=True,   # reduces RAM spike during load
)
model.eval()
print("Model loaded.")

# ── PDF → PIL Images (in-memory, no disk I/O) ────────────────────────────────
def pdf_to_images(pdf_path, dpi=DPI):
    """Convert each PDF page to a square PIL image (in RAM, no disk write)."""
    doc = fitz.open(pdf_path)
    images = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Make square (model handles square images more consistently)
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
                    "text": (
                        "Extract all visible text from this French document page. "
                        "Return plain text only, preserving the original structure."
                    ),
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
        images=[image],   # must be a list
        padding=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,          # greedy decoding — faster + more consistent
            temperature=None,         # disable sampling
            top_p=None,               # disable nucleus sampling
            use_cache=True,           # KV cache speeds up generation
        )

    # Decode only the newly generated tokens (skip the prompt)
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

    print(f"\nDone! Extracted text saved to: {OUTPUT_JSON}")
