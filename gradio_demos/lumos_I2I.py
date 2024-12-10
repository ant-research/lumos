import gradio as gr
import simple_parsing
import numpy as np
import torch
from torchvision.utils import make_grid
import random
import torch
from diffusers.models import AutoencoderKL
from lumos_diffusion import DPMS_INTER
from utils.download import find_model
import lumos_diffusion.model.dino.vision_transformer as vits
import torchvision.transforms as T
from lumos_diffusion.model.lumos import LumosI2I_XL_2
from utils import find_model

_TITLE = 'Lumos-I2I: Image Interpolation Generation'
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
    prompt_img1,
    prompt_img2,
    bsz,
    guidance_scale=4.5,
    num_inference_steps=20,
    seed=10,
    randomize_seed=True
):
    seed = int(randomize_seed_fn(seed, randomize_seed))
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    vae, dino, transform, model = models["vae"], models["vision_encoder"], models["transform"], models["diffusion"]
    prompt_img1 = transform(prompt_img1).unsqueeze(0)
    prompt_img2 = transform(prompt_img2).unsqueeze(0)
    prompt_imgs = torch.cat([prompt_img1, prompt_img2], dim=0)
    with torch.no_grad():
        caption_embs = dino(prompt_imgs.to(device))
        caption_embs = torch.nn.functional.normalize(caption_embs, dim=-1).unsqueeze(1).unsqueeze(1)
        caption_emb1 = caption_embs[0]
        caption_emb2 = caption_embs[-1]
        weights = np.arange(0, 1, 1/bsz).tolist()
        caption_embs = [caption_emb2 * wei + caption_emb1 * (1-wei) for wei in weights]
        caption_embs = torch.stack(caption_embs).to(device)
        bsz = caption_embs.shape[0]
        null_y = model.y_embedder.y_embedding[None].repeat(bsz, 1, 1)[:, None]
        z = torch.randn(1, 4, 32, 32, device=device).repeat(bsz, 1, 1, 1)
        model_kwargs = dict(mask=None)
        dpm_solver = DPMS_INTER(model.forward_with_dpmsolver,
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
            make_grid(output, nrow=output.shape[0] // 3, padding=3, pad_value=1).permute(1, 2, 0).numpy() * 255
        ).astype(np.uint8)
        step = num_inference_steps
        yield output, seed, gr.update(
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
    demo = gr.Blocks(css=css)
    with demo:
        with gr.Column(elem_id="col-container"):
            gr.Markdown('# ' + _TITLE)
            gr.Markdown("You can get various visual effects by adjusting the hyper-parameters in Advanced settings.")
            pid = gr.State()
            with gr.Row(equal_height=True):
                prompt_image1 = gr.Image(type="pil", label="Input Image 1")
                prompt_image2 = gr.Image(type="pil", label="Input Image 2")
            with gr.Row(equal_height=True):
                num_generation = gr.Slider(
                    value=12,
                    minimum=1,
                    maximum=100,
                    step=2,
                    label="Generation Num",
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
                randomize_seed = gr.Checkbox(label="Randomize seed", value=False)
                seed = gr.Slider(
                    value=137,
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    label="Random seed",
                )

            run_event = run_btn.click(
                fn=generate,
                inputs=[
                    prompt_image1,
                    prompt_image2,
                    num_generation,
                    guidance_scale,
                    num_inference_steps,
                    seed,
                    randomize_seed
                ],
                outputs=[
                    output_image,
                    seed,
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
            with gr.Row(equal_height=False):
                example_images_1 = ["asset/images/car/image_start.png", "asset/images/cat/image_start.JPG", "asset/images/folwer/image_start.png"]
                example_images_2 = ["asset/images/car/image_end.png", "asset/images/cat/image_end.JPG", "asset/images/folwer/image_end.png"]
                example = gr.Examples(
                    examples=[[t[0].strip(), t[-1].strip()] for t in zip(example_images_1, example_images_2)],
                    inputs=[prompt_image1, prompt_image2],
                )

        launch_args = {"server_port": int(args.port), "server_name": "0.0.0.0"}
        demo.queue(default_concurrency_limit=1).launch(**launch_args)


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser(description="Lumos Image Interpolation Generation Demo")
    parser.add_argument("--vae-pretrained", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--dino-type", type=str, default="vit_base")
    parser.add_argument("--dino-pretrained", type=str, default="./checkpoints/dino_vitbase16_pretrain.pth")
    parser.add_argument("--lumos-i2i-ckpt", type=str, default="./checkpoints/Lumos_I2I.pth")
    parser.add_argument("--port", type=int, default=19231)
    args = parser.parse_known_args()[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # setting models
    models = dict()
    ## autoencoder
    weight_dtype = torch.float32
    vae = AutoencoderKL.from_pretrained(args.vae_pretrained).cuda()
    vae.eval()
    vae.to(weight_dtype)
    models["vae"] = vae
    ## vision encoder 
    dino = vits.__dict__[args.dino_type](patch_size=16, num_classes=0).cuda()
    state_dict = torch.load(args.dino_pretrained, map_location="cpu")
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = dino.load_state_dict(state_dict, strict=False)
    del state_dict
    dino.eval()
    models["vision_encoder"] = dino
    ## transform for vision encoder
    transform = [
            T.Lambda(lambda img: img.convert('RGB')),
            T.Resize(224),  # Image.BICUBIC
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]

    transform = T.Compose(transform)
    models["transform"] = transform
    ## diffusion model
    model_kwargs={"window_block_indexes": [], "window_size": 0, 
                    "use_rel_pos": False, "lewei_scale": 1.0, 
                    "caption_channels": dino.embed_dim, 'model_max_length': 1}
    # build models
    image_size = 256
    latent_size = int(image_size) // 8
    model = LumosI2I_XL_2(input_size=latent_size, **model_kwargs).to(device)
    state_dict = find_model(args.lumos_i2i_ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(weight_dtype)
    models["diffusion"] = model
    
    demo(args)
