import torch
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from torchvision.utils import save_image
from torchvision import transforms as tfms
from PIL import Image
from torch.utils.data.dataloader import default_collate
from utils import _AVIDDataset
from utils import parse_args_and_config, norm
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse


def load_image(image_path, size, device):
    """从指定路径加载图像，调整大小并转换为适合模型的张量"""
    img = Image.open(image_path).convert('RGB')
    if size:
        img = img.resize((size,size))
    return tfms.functional.to_tensor(img).unsqueeze(0).to(device,dtype=torch.float16) * 2 - 1


def generate_latent(vae, x, size, batch_size, device):
    # 图像latent
    # latents = vae.encode(load_image(image_path, size, device))
    
    latents = vae.encode(x)
    latents = 0.18215 * latents.latent_dist.sample()
    # latents = latents.repeat(batch_size, 1, 1, 1)
    # scale the initial noise by the standard deviation required by the scheduler
    # latents = latents * noise_scheduler.init_noise_sigma # for DDIM, init_noise_sigma = 1.0
    return latents

def custom_collate_fn(batch):
    # 过滤掉所有空的张量
    batch = [item for item in batch if item[0].nelement() > 0]
    if not batch:
        return torch.tensor([]), []
    return default_collate(batch)

@torch.no_grad()
def pipe(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = args.device
    # model_id = args.diffusion_id
    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000
    # input_image = '/groups/generation_models/home/share/artifact/artifact/val/real/img3.jpg'
    load_size = args.loadSize
    

    guidance_scale = args.guidance_scale
    batch_size = args.batch_size
    t = args.selected_step

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device,dtype=torch.float16)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    model.float()
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
        # Labels to condition the model with (feel free to change):
    class_labels = [31,31,31,31]

    # Create sampling noise:
    n = len(class_labels)
    # z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    # z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
    

    

    dataset = _AVIDDataset(args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=int(args.num_threads), collate_fn=custom_collate_fn)
    for batch in tqdm(dataloader):
        x, save_paths = batch
        x = x.to(device).float()
        # print(x[0].type)
        # exit("here")
        latents = generate_latent(vae, x, load_size, batch_size, device)
        # 这里latens扩展2份，和text_embeddings对应，能一并计算unconditional prediction
        latent_model_input = torch.cat([latents] * 2)
        # latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t) # for DDIM, do nothing
        # print("Latent model input shape:", latent_model_input.shape)
        # print("Latent model input shape[0]:", latent_model_input.shape[0])
        # print("Text embeddings shape:", text_embeddings.shape)
        # print("text embeddings :", text_embeddings)
        # text_embeddings_tmp = text_embeddings.repeat(latent_model_input.shape[0]//2,1,1)
        # print("Latent model input shape:", latent_model_input.shape)
        # print("Text embeddings shape:", text_embeddings_tmp.shape)
        
        # print("text embeddings after repeat:", text_embeddings)
        
        # 使用UNet预测 latent noise
        # noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings_tmp).sample
        # Sample images:
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, latent_model_input.shape, latent_model_input, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
        
        # samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        # samples = vae.decode(samples / 0.18215).sample   
        # print("decode done") 
        # print(f'noise_pred:{noise_pred}')
        # 执行CFG
        # noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        # avid_noises = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        avid_noises = 1 / 0.18215 * samples
        # print(f'avid_noises = 1 / 0.18215 * avid_noises:{avid_noises}')
        images = vae.decode(avid_noises).sample
        # Save and display images:
        save_image(images, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))
        
            
        for idx, item in enumerate(images):
            # print(f'image saved:{item}')
            # print(f'image.shape:{item.shape}')
            save_image(item, save_paths[idx], normalize=True)
        # except Exception as e:
        #     print(f"Error processing image: {e}")
    
if __name__ == '__main__':
    print('***********************')
    args = parse_args_and_config()
    pipe(args)