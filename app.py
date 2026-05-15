from __future__ import annotations

import gradio as gr

from src.pipeline import (
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_PROMPT,
    AncientPaintingInpainter,
)


inpainter = AncientPaintingInpainter.from_env()


def restore(
    image,
    mask,
    prompt,
    negative_prompt,
    steps,
    guidance_scale,
    strength,
    seed,
    preserve_unmasked,
):
    if image is None:
        raise gr.Error("Please upload a damaged image.")
    if mask is None:
        raise gr.Error("Please upload a binary mask. White pixels are repaired.")
    return inpainter.restore(
        image=image,
        mask=mask,
        prompt=prompt or DEFAULT_PROMPT,
        negative_prompt=negative_prompt or DEFAULT_NEGATIVE_PROMPT,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        strength=float(strength),
        seed=int(seed),
        preserve_unmasked=bool(preserve_unmasked),
        blur_mask=1.0,
    )


with gr.Blocks(title="Ancient Painting Inpainting") as demo:
    gr.Markdown("# Ancient Painting Inpainting")
    with gr.Row():
        image = gr.Image(type="pil", label="Damaged image")
        mask = gr.Image(type="pil", label="Mask, white means repair")
        output = gr.Image(type="pil", label="Repaired image")
    prompt = gr.Textbox(value=DEFAULT_PROMPT, label="Prompt", lines=2)
    negative_prompt = gr.Textbox(
        value=DEFAULT_NEGATIVE_PROMPT, label="Negative prompt", lines=2
    )
    with gr.Row():
        steps = gr.Slider(10, 80, value=30, step=1, label="Steps")
        guidance_scale = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance")
        strength = gr.Slider(0.2, 1.0, value=1.0, step=0.05, label="Strength")
        seed = gr.Number(value=42, precision=0, label="Seed")
    preserve_unmasked = gr.Checkbox(value=True, label="Preserve unmasked pixels")
    run = gr.Button("Restore", variant="primary")
    run.click(
        restore,
        inputs=[
            image,
            mask,
            prompt,
            negative_prompt,
            steps,
            guidance_scale,
            strength,
            seed,
            preserve_unmasked,
        ],
        outputs=output,
        api_name="restore",
    )


if __name__ == "__main__":
    demo.launch()
