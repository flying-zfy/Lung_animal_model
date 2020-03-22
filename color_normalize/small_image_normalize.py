
###  对数据集中的图片做染色标准化处理，target目标图片选取的是： /cptjack/totem/Lung_animal_model/Train/Lung fibrosis(Ashcroft)/training/7/17_305_18_19_12.png

from __future__ import division
import stain_utils as utils
import stainNorm_Macenko
import cv2
import os
import numpy as np
from tqdm import tqdm 
import glob
from skimage import io


def contrast_img(img1, c=1.7, b=3):
    rows, cols, channels = img1.shape
    blank = np.zeros([rows, cols, channels], img1.dtype)
    dst = cv2.addWeighted(img1, c, blank, 1 - c, b)
    return dst
    
def colorN(img_path):
    img = utils.read_image(img_path)
    fit_img = utils.read_image('/cptjack/totem/zhaofeiyan/DataSet/data-ori/target.png')
    n = stainNorm_Macenko.Normalizer()
    n.fit(fit_img)
    t_img = n.transform(img)
    return t_img

if __name__ == '__main__':
    
    base_path = '/cptjack/totem/Lung_animal_model/Train/Lung fibrosis (Ashcroft)/training/ignore'
    save_path = '/cptjack/totem/zhaofeiyan/DataSet/data-ori/norm/ignore'
    img_path = glob.glob(os.path.join(base_path, "*.png"))
    img_path = sorted(img_path)
    for image in tqdm(img_path):
        image_name = os.path.basename(image).split('.')[0]
        img = colorN(image)
        io.imsave(os.path.join(save_path,f"{image_name}.png"),img)
