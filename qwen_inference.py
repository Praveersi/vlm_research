from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import config
from pathlib import Path

def analyze_image(image_path, text_prompt):
    # Load the model and processor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", 
        torch_dtype="auto", 
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    # Format the messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{Path(image_path).absolute()}"},
                {"type": "text", "text": text_prompt},
            ],
        }
    ]

    # Process inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    # Generate output
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

if __name__ == "__main__":
    image_path = config.FIGURES_DIR / "phase0_pairs.png"
    prompt = "Look at this grid of satellite images. The blue areas in the middle column represent water masks, and the red outlines in the right column show water boundaries. Can you describe the extent of flooding in these areas based on the images?"
    
    print(f"Analyzing {image_path} with Qwen2.5-VL-7B...")
    result = analyze_image(image_path, prompt)
    print("\nQwen2.5-VL Output:")
    print(result)
