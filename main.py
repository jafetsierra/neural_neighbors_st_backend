# Core Imports
import time
import random
from fastapi import FastAPI, File, UploadFile
import os
import matplotlib.pyplot as plt
from fastapi import Response
import json
from fastapi.middleware.cors import CORSMiddleware
# External Dependency Imports
from imageio import imwrite
import torch
import numpy as np
import requests
# Internal Project Imports
from pretrained.vgg import Vgg16Pretrained
from utils import misc as misc
from utils.misc import load_path_for_pytorch
from utils.stylize import produce_stylization

#FastAPI 
app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fix Random Seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def style_img_path(style):
    images = os.listdir('./img')
    if style in images:
        img_path = os.path.join('img',style)
    return img_path

def run(content_path=None,file=None, high_res=False,style_path=None,output_path=None,sz=512, alpha=0.75):
    # Interpret command line arguments
    max_scls = 4
    if high_res:
        max_scls = 5
        sz = 1024
    flip_aug = False
    content_loss = False
    misc.USE_GPU = True
    content_weight = 1. - alpha
    dont_colorize = False
    # Error checking for arguments
    # error checking for paths deferred to imageio
    assert (0.0 <= content_weight) and (content_weight <= 1.0), "alpha must be between 0 and 1"
    assert torch.cuda.is_available() or (not misc.USE_GPU), "attempted to use gpu when unavailable"

    # Define feature extractor
    cnn = misc.to_device(Vgg16Pretrained())
    phi = lambda x, y, z: cnn.forward(x, inds=y, concat=z)

    # Load images
    print("loaging content and style")
    content_im_orig = misc.to_device(load_path_for_pytorch(im_file=file, target_size=sz)).unsqueeze(0)
    style_im_orig = misc.to_device(load_path_for_pytorch(style_path, target_size=sz)).unsqueeze(0)

    # Run Style Transfer
    print("starting stylized process")
    torch.cuda.synchronize()
    start_time = time.time()
    output = produce_stylization(content_im_orig, style_im_orig, phi,
                                max_iter=200,
                                lr=1e-3,
                                content_weight=content_weight,
                                max_scls=max_scls,
                                flip_aug=flip_aug,
                                content_loss=content_loss,
                                dont_colorize=dont_colorize)
    torch.cuda.synchronize()
    print('Done! total time: {}'.format(time.time() - start_time))

    # Convert from pyTorch to numpy, clip to valid range
    new_im_out = np.clip(output[0].permute(1, 2, 0).detach().cpu().numpy(), 0., 1.)

    # Save stylized output
    save_im = (new_im_out * 255).astype(np.uint8)
    imwrite(output_path, save_im)

    # Free gpu memory in case something else needs it later
    if misc.USE_GPU:
        torch.cuda.empty_cache()
    return save_im

    
@app.post("/")
def file_process(file: UploadFile, style_name:str):
    result = run(file=file.file, style_path=style_img_path(style_name),output_path='./outputs/output.jpg',alpha=0.7)
    
    url = "https://api.imgbb.com/1/upload"
    payload = {}
    files = {
        'image': ('./outputs/output.jpg', open('./outputs/output.jpg', 'rb')),
    }
    headers = {
    'key': '60a973fe05ef2eb8630752d9f05147a6',
    'expiration': '120'
    }
    response = requests.request("POST", url, params=headers, files=files)
    image_url = dict(response.json())['data']['url']
    return Response(image_url)