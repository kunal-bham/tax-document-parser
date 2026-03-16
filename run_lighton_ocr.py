import torch
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor
from PIL import Image


def main() -> None:
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    dtype = torch.float32 if device == "mps" else torch.bfloat16

    print(f"Using device={device}, dtype={dtype}")

    model = LightOnOcrForConditionalGeneration.from_pretrained(
        "lightonai/LightOnOCR-2-1B", torch_dtype=dtype
    ).to(device)
    processor = LightOnOcrProcessor.from_pretrained("lightonai/LightOnOCR-2-1B")

    image_path = "/home/bhamk/tax-document-parser/tax-document-parser/debug_page1.png"
    image = Image.open(image_path).convert("RGB")

    conversation = [{"role": "user", "content": [{"type": "image", "image": image}]}]

    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {
        k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
        for k, v in inputs.items()
    }

    output_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    output_text = processor.decode(generated_ids, skip_special_tokens=True)
    print("\n=== OCR OUTPUT ===")
    print(output_text)


if __name__ == "__main__":
    main()

