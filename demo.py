import argparse
import os
import matplotlib.pyplot as plt

from model import *
from model.eccv16 import eccv16
from model.sig17 import siggraph17
from model.util import *

def load_input_image(img_path):
    return load_img(img_path)

def get_input_image_path(img_dir, img_index):
    return os.path.join(img_dir, f'{img_index}.jpg')

parser = argparse.ArgumentParser()
parser.add_argument('--img_indices', type=int, nargs='+', default=[1, 2, 3], help='Indices of the input images (1.jpg, 2.jpg, ...)')
parser.add_argument('--img_dir', type=str, default='landscape', help='Directory where the input images are located')
opt = parser.parse_args()

# Load colorizers
colorizer_eccv16 = eccv16().eval()
colorizer_siggraph17 = siggraph17().eval()

plt.figure(figsize=(12, 8))

for i, img_index in enumerate(opt.img_indices, start=1):
    # Default size to process images is 256x256
    # Grab L channel in both original ("orig") and resized ("rs") resolutions
    img_path = get_input_image_path(opt.img_dir, img_index)
    img = load_input_image(img_path)
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))

    # Colorizer outputs 256x256 ab map
    # Resize and concatenate to original L channel
    img_bw = postprocess_tens(tens_l_orig, torch.cat((torch.zeros_like(tens_l_orig), torch.zeros_like(tens_l_orig)), dim=1))
    out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
    out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

    plt.subplot(4, 3, i)
    plt.imshow(img)
    plt.title(f'Original #{img_index}')
    plt.axis('off')

    plt.subplot(4, 3, i + 3)
    plt.imshow(img_bw)
    plt.title(f'Grayscale #{img_index}')
    plt.axis('off')
    

    plt.subplot(4, 3, i + 6)
    plt.imshow(out_img_siggraph17)
    plt.title(f'SIGGRAPH 17 for #{img_index}')
    plt.axis('off')
    
    plt.subplot(4, 3, i + 9)
    plt.imshow(out_img_eccv16)
    plt.title(f'ECCV16 17 for #{img_index}')
    plt.axis('off')
    

plt.tight_layout()
plt.show()