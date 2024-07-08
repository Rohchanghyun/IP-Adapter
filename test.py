import os
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from PIL import Image

from ip_adapter import IPAdapter

base_model_path = "CompVis/stable-diffusion-v1-4" 
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "h94/IP-Adapter"
ip_ckpt = "fashion_image/checkpoint-30000/ip-adapter.bin"
device = "cuda"

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

# load SD pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
)

image_encoder = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder="models/image_encoder")

# load ip-adapter
ip_model = IPAdapter(pipe, image_encoder, ip_ckpt, device)

# load SD Img2Img pipe
del pipe, ip_model
torch.cuda.empty_cache()
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
)

# read image prompt
image = Image.open("assets/images/1.png")
g_image = Image.open("assets/images/2.jpg")


# generate
images = ip_model.generate(pil_image=image, num_samples=4, num_inference_steps=50, seed=42, image=g_image, strength=0.6)
grid = image_grid(images, 1, 4)

# grid를 이미지로 저장
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
grid_path = os.path.join(output_dir, "grid_image.png")
grid.save(grid_path)

print(f"Grid image saved at {grid_path}")
