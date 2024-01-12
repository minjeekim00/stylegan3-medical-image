import os
import argparse
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from glob import glob
from tqdm import tqdm

from loader.utils import window
from loader.utils import apply_window
from loader.utils import dcm_to_array
from loader.utils import bodyct_dcm_to_pil
from loader.utils import brainct_dcm_to_pil


def get_args_parser():
    parser = argparse.ArgumentParser('Osteo Classification Train and Test script', add_help=False)

    # Dataset parameters
    parser.add_argument('--src-dir', default="/mnt/dataset/Synthesis_Study/2022/AbdomenCT", type=str, help='dataset folder dirname')
    parser.add_argument('--dst-dir', default="'/mnt/dataset/Synthesis_Study/2022/AbdomenCT_png'", type=str, help='dataset folder dirname')
    parser.add_argument('--modality',  choices=['chestxray', 'abdomenct', 'brainct'], required=True)
    parser.add_argument('--resolution',  default=512, type=int)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Osteo Classification training and evaluation script', parents=[get_args_parser()])
    args   = parser.parse_args()


    
    files = glob(f"{src_dir}/**/*.dcm"), recursive=True)
    print(f"{len(files)} dicom files are found in {src_dir}")


    window_ranges = {'brainct': (0, 80),
                     'abdomenct': (-140, 210)}
    w_ragne = window_ranges[args.modality]

    if args.modality == 'abdomenct':

        for file in tqdm(files):
            pil_img = bodyct_dcm_to_pil(file, resolution, w_range=w_range, body_only=True)
            
            dst = file.replace(src_dir, dst_dir)
            os.makedirs(os.path.split(dst)[0])
            dst = dst.replace('.dcm', '.png')
                        
            if os.path.exists(dst):
                continue
            pil_img.save(dst, quality=100)


    elif args.modality == 'brainct':

        for file in tqdm(files):
            pil_img = brainct_dcm_to_pil(file, resolution, w_range=w_range, body_only=True)

            dst = file.replace(src_dir, dst_dir)
            os.makedirs(os.path.split(dst)[0])
            dst = dst.replace('.dcm', '.png')

            if os.path.exists(dst):
                continue
            pil_img.save(dst, quality=100)







