import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, gaussian, sobel
from skimage.segmentation import active_contour

def run_comparison(img):
    # 1. Global Otsu
    # 排除全黑背景計算門檻
    valid_pixels = img[img > 0]
    thresh_val = threshold_otsu(valid_pixels) if len(valid_pixels) > 0 else 0
    _, res_otsu = cv2.threshold(img, thresh_val, 255, cv2.THRESH_BINARY)

    # 2. Adaptive Threshold
    res_adaptive = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 31, 9)  ## mofify block size and C 
    res_adaptive_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                         cv2.THRESH_BINARY, 41, 11)
    
    return res_otsu, res_adaptive, res_adaptive_mean

def main():
    radiograph_folder = '/home/chienyo2/BIOE484_teeth_seg/484TeethSegmentation/Radiographs'
    image_list = image_list = [f for f in os.listdir(radiograph_folder) if f.lower().endswith(('.jpg'))]
    os.makedirs('/home/chienyo2/BIOE484_teeth_seg/classical_seg_otsu/', exist_ok=True)
    os.makedirs('/home/chienyo2/BIOE484_teeth_seg/classical_seg_adaptive_gaussian/', exist_ok=True)
    os.makedirs('/home/chienyo2/BIOE484_teeth_seg/classical_seg_adaptive_mean/', exist_ok=True)
    print(f'Found {len(image_list)} images in {radiograph_folder}')
    for image_name in image_list:
        image_path = os.path.join(radiograph_folder, image_name) 
        print(f'Processing {image_path}...')
        img_raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img_blur = cv2.GaussianBlur(img_raw, (5, 5), 0)
        img_prep = clahe.apply(img_blur)
        otsu, adaptive_gaussian, adaptive_mean = run_comparison(img_prep)

        cv2.imwrite(f'/home/chienyo2/BIOE484_teeth_seg/classical_seg_otsu/{image_name}', otsu)
        cv2.imwrite(f'/home/chienyo2/BIOE484_teeth_seg/classical_seg_adaptive_gaussian/{image_name}', adaptive_gaussian)
        cv2.imwrite(f'/home/chienyo2/BIOE484_teeth_seg/classical_seg_adaptive_mean/{image_name}', adaptive_mean)


if __name__ == "__main__":
    main()

