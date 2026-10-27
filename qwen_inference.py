from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import config
from pathlib import Path
import json

def analyze_image(model, processor, image_path, text_prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{Path(image_path).absolute()}"},
                {"type": "text", "text": text_prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

if __name__ == "__main__":
    # Load model and processor
    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", 
        torch_dtype="auto", 
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    print("Model loaded successfully.\n")

    # Load the metadata
    meta_path = config.PROCESSED_DIR / "metadata.json"
    with open(meta_path, "r") as f:
        metadata = json.load(f)
        
    # Get the first available chip ID to test
    test_chip_id = list(metadata.keys())[0]
    chip_data = metadata[test_chip_id]
    
    # Path to the specific image chip
    image_path = config.ROOT / chip_data["image_path"]
    
    print(f"--- Analyzing Chip: {test_chip_id} ---")
    print(f"Ground Truth Water Fraction: {chip_data['stats']['water_fraction'] * 100}%")
    print(f"Image File: {image_path}\n")

    # Test 1: Free Text
    print("Test 1: Free Text Report")
    ans_free = analyze_image(model, processor, image_path, chip_data['prompts']['free_text'])
    print(f"Output:\n{ans_free}\n")
    
    # Test 2: JSON
    print("Test 2: Structured JSON")
    ans_json = analyze_image(model, processor, image_path, chip_data['prompts']['structured_json'])
    print(f"Output:\n{ans_json}\n")

    # Test 3: POPE (Yes/No)
    print("Test 3: POPE (Yes/No)")
    ans_pope = analyze_image(model, processor, image_path, chip_data['prompts']['pope'])
    print(f"Output:\n{ans_pope}\n")
