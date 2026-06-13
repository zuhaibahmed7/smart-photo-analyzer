import spaces
import gradio as gr
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, pipeline
from qwen_vl_utils import process_vision_info
import torch

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    min_pixels=256*28*28,
    max_pixels=512*28*28,
)

from transformers import MarianMTModel, MarianTokenizer
trans_model_name = "Helsinki-NLP/opus-mt-en-ur"
trans_tokenizer = MarianTokenizer.from_pretrained(trans_model_name)
trans_model = MarianMTModel.from_pretrained(trans_model_name)
if torch.cuda.is_available():
    trans_model = trans_model.to("cuda")
trans_model.eval()

PROMPTS = {
    "🔍 General Description": "Describe this image in detail. What do you see?",
    "🧾 Read Text / Receipt": "Extract and list all text visible in this image. If it's a receipt, list all items and prices.",
    "🌿 Identify Plant / Food": "What plant, crop, or food is this? Describe its condition and any issues.",
    "📄 Explain Document": "What does this document say? Summarize the key information clearly.",
    "🩺 Health / Medical Image": "Describe what you see in this image in simple, non-alarming terms.",
}

@spaces.GPU(duration=120)
def analyze_image(image, task):
    if image is None:
        return "⚠️ Please upload an image first.", ""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPTS[task]},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    result = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return result, result

@spaces.GPU(duration=60)
def translate_to_urdu(english_text):
    if not english_text or english_text.startswith("⚠️"):
        return "⚠️ Please analyze an image first."
    try:
        sentences = [s.strip() for s in english_text.split('.') if s.strip()]
        translated = []
        for sentence in sentences:
            tokens = trans_tokenizer([sentence], return_tensors="pt", padding=True)
            if torch.cuda.is_available():
                tokens = {k: v.to("cuda") for k, v in tokens.items()}
            with torch.no_grad():
                output = trans_model.generate(**tokens)
            translated.append(trans_tokenizer.decode(output[0], skip_special_tokens=True))
        return ' '.join(translated)
    except Exception as e:
        return f"Translation error: {str(e)}"

def clear_all():
    """Clears image, result and state — also cancels any running job."""
    return None, "", "", "Ready — upload an image to begin."

def image_removed():
    """Auto-clear result when image is deleted."""
    return "", "", "🗑️ Image removed. Upload a new image to analyze."

def set_status_analyzing():
    return "⏳ Analyzing image..."

def set_status_translating():
    return "🌐 Translating to Urdu..."

def set_status_done(result, state):
    return result, state, "✅ Done!"

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="orange"),
    title="Smart Photo Analyzer",
) as demo:

    gr.HTML("""
    <div style='text-align:center;padding:1rem 0 0.5rem'>
      <h1 style='font-size:1.8rem;font-weight:700'>📷 Smart Photo Analyzer</h1>
      <p style='color:#666;font-size:0.95rem;margin-top:4px'>
        Upload any image — get instant analysis in English or Urdu
      </p>
    </div>
    """)

    english_state = gr.State("")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            task_input = gr.Dropdown(
                choices=list(PROMPTS.keys()),
                value="🔍 General Description",
                label="What do you want to know?",
            )
            with gr.Row():
                analyze_btn = gr.Button("Analyze 🔍", variant="primary", scale=3)
                stop_btn = gr.Button("Stop ⛔", variant="stop", scale=1)
            with gr.Row():
                translate_btn = gr.Button("Translate to Urdu 🌐", variant="secondary", scale=3)
                clear_btn = gr.Button("Clear 🗑️", variant="secondary", scale=1)

        with gr.Column():
            status_box = gr.Textbox(
                value="Ready — upload an image to begin.",
                label="Status",
                lines=1,
                interactive=False,
            )
            output = gr.Textbox(
                label="Result",
                lines=12,
                placeholder="Your analysis will appear here...",
            )

    gr.HTML("<p style='text-align:center;font-size:0.75rem;color:#999;margin-top:8px'>Powered by Qwen2-VL-2B · Helsinki-NLP · Build Small Hackathon 2026</p>")

    # Analyze click — show status, run, update status when done
    analyze_event = analyze_btn.click(
        fn=lambda: "⏳ Analyzing image — please wait...",
        inputs=None,
        outputs=status_box,
    ).then(
        fn=analyze_image,
        inputs=[image_input, task_input],
        outputs=[output, english_state],
    ).then(
        fn=lambda r: ("✅ Done! Click 'Translate to Urdu' to translate." if not r.startswith("⚠️") else "⚠️ No image uploaded."),
        inputs=output,
        outputs=status_box,
    )

    # Stop button — cancels the running analyze job
    stop_btn.click(
        fn=lambda: "⛔ Stopped.",
        inputs=None,
        outputs=status_box,
        cancels=[analyze_event],
    )

    # Translate click
    translate_btn.click(
        fn=lambda: "🌐 Translating...",
        inputs=None,
        outputs=status_box,
    ).then(
        fn=translate_to_urdu,
        inputs=[english_state],
        outputs=output,
    ).then(
        fn=lambda: "✅ Translation done!",
        inputs=None,
        outputs=status_box,
    )

    # Clear button — clears everything and cancels any running job
    clear_btn.click(
        fn=clear_all,
        inputs=None,
        outputs=[image_input, output, english_state, status_box],
        cancels=[analyze_event],
    )

    # Auto-clear result when image is removed by user
    image_input.clear(
        fn=image_removed,
        inputs=None,
        outputs=[output, english_state, status_box],
        cancels=[analyze_event],
    )

demo.launch()