
import os
import yaml
import argparse
from models import DiT_models

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torchvision.io import read_image
from torchvision.transforms.functional import InterpolationMode


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def parse_args_and_config():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--config", type=str, default="config.yaml", help="Name of the config, under ./dnf/config")
    parser.add_argument("--dataroot", type=str, default='./dataset', help='The path to dataset')
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument( "--diffusion_id", type=str, default="/groups/generation_models/home/share/models/runwayml--stable-diffusion-v1-5")
    parser.add_argument( "--postfix", type=str, default="_avid")
    parser.add_argument('--rz_interp', default='bicubic')
    parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
    
    parser.add_argument(
        '--gpu_ids', type=str, default='0',
        help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU'
    )
    
    parser.add_argument('--prompt', default='')
    parser.add_argument('--n_prompt', default='')
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    # parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--selected_step", type=int, default=0)

    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")

    args = parser.parse_args()

    # with open(os.path.join("./dnf/configs", args.config), "r") as f:
    #     config = yaml.safe_load(f)
    # config = dict2namespace(config)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    device = torch.device('cuda:{}'.format(args.gpu_ids[0])) if args.gpu_ids else torch.device('cpu')
    print(f"[Device]: {device}")
    args.device = device

    return args#, config

rz_dict = {'bilinear': InterpolationMode.BILINEAR,
        'bicubic': InterpolationMode.BICUBIC,
        'lanczos': InterpolationMode.LANCZOS,
        'nearest': InterpolationMode.NEAREST}


class _AVIDDataset(datasets.ImageFolder):
    def __init__(self, opt):
        super().__init__(opt.dataroot)
        
        self.root = opt.dataroot
        self.save_root =opt.dataroot + opt.postfix
        os.makedirs(self.save_root, exist_ok=True)
        print(f"[AVID Dataset]: From {self.root} to {self.save_root}")
        self.paths = []
        for foldername, _, fns in os.walk(self.root):
            if not os.path.exists(foldername.replace(self.root, self.save_root)):
                os.mkdir(foldername.replace(self.root, self.save_root))
            for fn in fns:
                path = os.path.join(foldername, fn)
                if not os.path.exists(path.replace(self.root, self.save_root) ):
                    self.paths.append(path)
                
        rz_func = transforms.Resize((opt.loadSize, opt.loadSize), interpolation=rz_dict[opt.rz_interp])
        aug_func = transforms.Lambda(lambda img: img)
        
        self.transform = transforms.Compose([
            rz_func,
            transforms.Lambda(lambda img: (img / 255.0) * 2.0 - 1.0),  # 归一化处理
            aug_func, 
        ])
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        
        path = self.paths[index]
        save_path = path.replace(self.root, self.save_root) 
        try:
            sample = read_image(path).to(torch.float16)
            
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return torch.empty(0), save_path  # Return an empty tensor if the image cannot be loaded

        if sample.shape[0] == 1:  
            sample = torch.cat([sample] * 3, dim=0)
        elif sample.shape[0] == 4:  
            sample = sample[:3, :, :]
            
        # print(f'debug-image:{sample}')
            
        sample = self.transform(sample)
        # print(f'debug-image:{sample}')
        return sample, save_path
    

def norm(x):
    return (x - x.min()) / (x.max() - x.min())