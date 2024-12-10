import gradio as gr
import simple_parsing
import numpy as np
import torch
from torchvision.utils import make_grid
import random
import torch
from diffusers.models import AutoencoderKL
from lumos_diffusion import DPMS
from utils.download import find_model
from lumos_diffusion.model.t5 import T5Embedder
from lumos_diffusion.model.lumos import LumosT2IMS_XL_2
from utils import find_model, get_closest_ratio, ASPECT_RATIO_1024_TEST

_TITLE = 'Lumos-T2I: Zero-shot Text to Image'
MAX_SEED = 2147483647

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def dividable(n):
    for i in range(int(np.sqrt(n)), 0, -1):
        if n % i == 0:
            break
    return i, n // i


def stop_run():
    return (
        gr.update(value="Run", variant="primary", visible=True),
        gr.update(visible=False),
    )

def generate(
    height=1024,
    width=1024,
    prompt="a chair",
    guidance_scale=4.5,
    num_inference_steps=250,
    seed=10,
    randomize_seed=True
):
    seed = int(randomize_seed_fn(seed, randomize_seed))
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    bsz = 1
    vae, t5, model = models["vae"], models["language_encoder"], models["diffusion"]
    prompt = prompt.strip() if prompt.endswith('.') else prompt
    close_hw, close_ratio = get_closest_ratio(height, width, ratios=ASPECT_RATIO_1024_TEST)
    output_comment = f"Convert Height: {height}, Width: {width} to [{close_hw[0]}, {close_hw[1]}]."
    hw, ar = torch.tensor([close_hw], dtype=torch.float, device=device), torch.tensor([[float(close_ratio)]], device=device)
    latent_size_h, latent_size_w = int(hw[0, 0] // 8), int(hw[0, 1] // 8)
    prompts = [prompt] * bsz
    with torch.no_grad():
        caption_embs, emb_masks = t5.get_text_embeddings(prompts)
        caption_embs = caption_embs.float()[:, None]
        null_y = model.y_embedder.y_embedding[None].repeat(bsz, 1, 1)[:, None]
        z = torch.randn(bsz, 4, latent_size_h, latent_size_w, device=device)
        model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
        dpm_solver = DPMS(model.forward_with_dpmsolver,
                            condition=caption_embs,
                            uncondition=null_y,
                            cfg_scale=guidance_scale,
                            model_kwargs=model_kwargs)
        output = dpm_solver.sample(
                z,
                steps=num_inference_steps,
                order=2,
                skip_type="time_uniform",
                method="multistep")
        output = vae.decode(output / 0.18215).sample
        output = torch.clamp(output * 0.5 + 0.5, min=0, max=1).cpu()
        output = (
            make_grid(output, nrow=dividable(bsz)[0]).permute(1, 2, 0).numpy() * 255
        ).astype(np.uint8)
        step = num_inference_steps
        yield output, seed, close_hw[0], close_hw[1], gr.update(
            value="Run",
            variant="primary",
            visible=(step == num_inference_steps),
        ), gr.update(
            value="Stop", variant="stop", visible=(step != num_inference_steps)
        )


def demo(args):
    css = """
    #col-container {
        margin: 0 auto;
        max-width: 640px;
    }
    """
    example_texts = open("asset/samples.txt").readlines()
    demo = gr.Blocks(css=css)
    with demo:
        with gr.Column(elem_id="col-container"):
            gr.Markdown('# ' + _TITLE)
            pid = gr.State()
            with gr.Row(equal_height=True):
                prompt_input = gr.Text(
                    label="Prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your prompt",
                    container=False,
                    scale=5
                )
                run_btn = gr.Button(value="Run", variant="primary", scale=1)
                stop_btn = gr.Button(value="Stop", variant="stop", visible=False)
            with gr.Row(equal_height=False):
                output_image = gr.Image(value=None, label="Output image")
            with gr.Accordion(
                "Advanced settings", open=False, elem_id="config-accordion"
            ):
                with gr.Row(equal_height=False):
                    num_inference_steps = gr.Slider(
                        value=20,
                        minimum=1,
                        maximum=2000,
                        step=1,
                        label="# of steps",
                    )
                    guidance_scale = gr.Slider(
                    value=4.5,
                    minimum=0.0,
                    maximum=50,
                    step=0.1,
                    label="Guidance scale",
                    )   
                with gr.Row(equal_height=False):
                    height = gr.Slider(
                    value=1024,
                    minimum=512,
                    maximum=2048,
                    step=32,
                    label="Height",
                    )
                    width = gr.Slider(
                    value=1024,
                    minimum=512,
                    maximum=2048,
                    step=32,
                    label="Width",
                    )
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                seed = gr.Slider(
                    value=10,
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    label="Random seed",
                )

            run_event = run_btn.click(
                fn=generate,
                inputs=[
                    height,
                    width,
                    prompt_input,
                    guidance_scale,
                    num_inference_steps,
                    seed,
                    randomize_seed
                ],
                outputs=[
                    output_image,
                    seed,
                    height,
                    width,
                    run_btn,
                    stop_btn,
                ],
            )

            stop_btn.click(
                fn=stop_run,
                outputs=[run_btn, stop_btn],
                cancels=[run_event],
                queue=False,
            )

            example0 = gr.Examples(
                examples=[[t.strip()] for t in example_texts],
                inputs=[prompt_input],
            )

        launch_args = {"server_port": int(args.port), "server_name": "0.0.0.0"}
        demo.queue(default_concurrency_limit=1).launch(**launch_args)


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser(description="Lumos Text to Image Generation Demo")
    parser.add_argument("--vae-pretrained", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--t5-path", type=str, default="./checkpoints/")
    parser.add_argument("--lumos-t2i-ckpt", type=str, default="./checkpoints/Lumos_T2I.pth")
    parser.add_argument("--port", type=int, default=19231)
    args = parser.parse_known_args()[0]
    if torch.cuda.is_available():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # setting models
        models = dict()
        ## autoencoder
        weight_dtype = torch.float16
        vae = AutoencoderKL.from_pretrained(args.vae_pretrained).cuda()
        vae.eval()
        models["vae"] = vae
        ## language encoder 
        t5 = T5Embedder(device="cuda", local_cache=True, cache_dir=args.t5_path, torch_dtype=torch.float)
        models["language_encoder"] = t5
        ## diffusion model
        model_kwargs={"window_block_indexes": [], "window_size": 0,
                        "use_rel_pos": False, "lewei_scale": 2.0}
        # build models
        image_size = 1024
        latent_size = int(image_size) // 8
        model = LumosT2IMS_XL_2(input_size=latent_size, **model_kwargs).to(device)
        state_dict = find_model(args.lumos_t2i_ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        model.eval()
        model.to(weight_dtype)
        models["diffusion"] = model
    else:
        raise ValueError("This Demo need gpu")

    demo(args)
