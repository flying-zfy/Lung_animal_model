
"""
@author: zhaofy

"""

from __future__ import division
import openslide as opsl
import numpy as np
import cv2
import os
from skimage import io 
from keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm 
import glob

import stain_utils as utils
import stainNorm_Macenko
 
# 指定显卡
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gpu_options=tf.GPUOptions(allow_growth=True)
sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def gray_binary(img,thresh,show = False):
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray,(5,5),0)
    binary = cv2.threshold(blurred,thresh,255,cv2.THRESH_BINARY_INV)[1]

    def binary_threshold(threshold):
        binary = cv2.threshold(blurred,threshold,255,cv2.THRESH_BINARY_INV)[1]
        cv2.imshow("binary image",binary)
        
    if show:
        window_name = "binary image"
        cv2.namedWindow(window_name, 0)
        cv2.resizeWindow(window_name, 640, 480)
        cv2.createTrackbar("binary threshold", window_name, 150, 255, binary_threshold)
        binary_threshold(150)
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
            
    return binary

def erode_dilate(binary, erode_iter=6, dilate_iter=9, kernel_size=(5, 5), show=False):
    morphology = binary
    if show:
        window_name = "errode dilate"
        cv2.namedWindow(window_name, 0)
        cv2.resizeWindow(window_name, 640, 480)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    while show:
        cv2.imshow(window_name, morphology)
        key = cv2.waitKey(1) & 0xFF
        if ord('e') == key:
            morphology = cv2.erode(morphology, kernel, iterations=1)
            print('erode')
        if ord('d') == key:
            morphology = cv2.dilate(morphology, kernel, iterations=1)
            print('dilate')
        if ord('r') == key:
            morphology = binary
            print('reset threshold image')
        if ord('q') == key:
            break
            
    cv2.destroyAllWindows()
    morphology = cv2.erode(morphology, kernel, iterations=erode_iter)
    morphology = cv2.dilate(morphology, kernel, iterations=dilate_iter)

    return morphology

def find_contours(morphology):
    image, cnts, hierarchy = cv2.findContours(morphology.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts

def get_mask(img):
    
    #  img：ndpi图片在四级下的缩略图
    #  该方法与前面三个函数结合使用，返回ndpi整张图片四级下的mask，预测大图时用来区分组织区域和背景    
    
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    v = (v).astype(np.uint8)
    v = v + 70            #  调整图片亮度，用于更好识别组织区域  
    v[v < 50] = 255
    hsv1 = cv2.merge((h,s,v))
    img1= cv2.cvtColor(hsv1,cv2.COLOR_HSV2RGB) 
    img1 = cv2.cvtColor(img1,cv2.COLOR_RGB2BGR)
    binary = gray_binary(img1,thresh = 230,show = False)
    morphology = erode_dilate(binary, erode_iter=0, dilate_iter=3, show=False)
    cnts = find_contours(morphology)
    mask = np.zeros((img1.shape[0], img1.shape[1])).astype(img1.dtype)
    color = [2]
    mask = cv2.fillPoly(mask, cnts, color)
    
    return mask

def colorN(img):
    
    #  染色标准化处理，选取目标图片target.png作为参考图，调用stainNorm_Macenko.Normalizer()方法

    fit_img = utils.read_image('/cptjack/totem/zhaofeiyan/DataSet/data-ori/target.png')
    n = stainNorm_Macenko.Normalizer()
    n.fit(fit_img)
    t_img = n.transform(img)
    return t_img

def contrast_img(img1, c=1.7, b=3):
    
    #  图片对比度调整
    
    rows, cols, channels = img1.shape
    blank = np.zeros([rows, cols, channels], img1.dtype)
    dst = cv2.addWeighted(img1, c, blank, 1-c, b)
    return dst


def get_out_img(model,svs,mask,patch_size,level):

    #  model: 训练好的模型
    #  svs:  ndpi图片的路径
    #  mask: 获取到的整个组织区域的掩码图
    #  patch_size:  预测时滑动的步长  
    #  level:  层级
    #  返回的是一个预测结果矩阵
    
    slide = opsl.open_slide(svs)
    level_downsamples = slide.level_downsamples[level]
    Wh = np.zeros((len(slide.level_dimensions),2))
    for i in range (len(slide.level_dimensions)):
        Wh[i,:] = slide.level_dimensions[i]
        Ds = np.zeros((len(slide.level_downsamples),2))
    for i in range (len(slide.level_downsamples)):
        Ds[i,0] = slide.level_downsamples[i]
        Ds[i,1] = slide.get_best_level_for_downsample(Ds[i,0]) 

    w_count = int(Wh[0,0] // patch_size)
    h_count = int(Wh[0,1] // patch_size)
    out_img = np.zeros([h_count,w_count])
    count = 0
    for x in tqdm(range (w_count)):
        for y in range (h_count):
            w0 = x * patch_size
            h0 = y * patch_size
            bottom = int(y * patch_size / level_downsamples)
            top = bottom + int(patch_size / level_downsamples) -1
            left = int(x * patch_size / level_downsamples)
            right = left + int(patch_size / level_downsamples) -1
            if np.sum(mask[bottom : top,left : right ] > 0) > 0.75 * (patch_size / level_downsamples)**2:
                subHIC = np.array(slide.read_region((w0, h0), 0, (patch_size, patch_size)))[:,:,:3]
                subHIC = contrast_img(subHIC)
                subHIC = cv2.cvtColor(subHIC,cv2.COLOR_BGR2RGB)
                subHIC = cv2.cvtColor(subHIC,cv2.COLOR_BGR2RGB)
                rgb_s1 = (abs(subHIC[:,:,0] -107) >= 93) & (abs(subHIC[:,:,1] -107) >= 93) & (abs(subHIC[:,:,2] -107) >= 93)
                if np.sum(rgb_s1)<=(patch_size * patch_size ) * 0.98:
                    # 模型中的图片输入格式为(1,224,224,3),因此做resize处理,这里采用最近邻插值方法，因为训练的时候keras的resize方法默认为最近邻插值方法
                    subHIC1 = cv2.resize(subHIC, (224,224), interpolation = cv2.INTER_NEAREST)     
                    subHIC2 = colorN(subHIC1)                 # 染色标准化处理
                    subHIC3 = np.array([subHIC2])
                    subHIC3 = subHIC3.reshape(1,224,224,3)
                    prob = model.predict(subHIC3 / 255.0)     #  与训练时的预处理操作保持一致
                    prob1 = int(round(prob[0][1]))            #  只对非ignore(no-ignore)的概率进行处理，并保存下来，最终结果显示的是非ignore的区域，没有预测值的部分属于ignore区域；
                    out_img[y,x] = prob1
                    count = count + 1
    slide.close()
    out_img = cv2.resize(out_img, (int(w_count * patch_size /Ds[level,0]), int(h_count * patch_size /Ds[level,0])), interpolation=cv2.INTER_AREA)
    out_img = cv2.copyMakeBorder(out_img,0,int(Wh[level,1]-out_img.shape[0]),0,int(Wh[level,0]-out_img.shape[1]),cv2.BORDER_REPLICATE)
    
    return out_img

def colormap(svs_im,out_img,title,save_dir):
    
    #  该方法用于将预测结果用热力图的形式显示出来并保存
    #  svs_im: ndpi四级下的缩略图
    #  out_img: 预测的结果矩阵
    #  title: 标题
    #  save_dir: 保存路径
    
    plt_size = (svs_im.size[0] // 100, svs_im.size[1] //100)
    flg, ax = plt.subplots(figsize = plt_size, dpi =100)      ###  实例化两个子图
    matrix = out_img.copy()
    matrix = cv2.resize(matrix, svs_im.size, interpolation = cv2.INTER_AREA)
    cax = ax.imshow(matrix, cmap = plt.cm.jet, alpha = 0.45)   ### cmap ：颜色图谱，jet : 蓝-青-黄-红
    svs_im_npy = np.array(svs_im.convert('RGBA'))
    svs_im_npy[:,:][matrix[:,:] > 0] = 0 
    ax.imshow(svs_im_npy)
    max_matrix_value = matrix.max()    #获取矩阵最大值，用来方便标识colorbar的最高刻度值
    plt.colorbar(cax, ticks = np.linspace(0, max_matrix_value, 15, endpoint = True))    #设置colorbar，0为最低刻度，max_matrix_value为最高刻度，15为分成15个刻度
    ax.set_title(title, fontsize = 40)   #设置图片标题
    plt.axis('off')

    if not os.path.isdir(save_dir):os.makedirs(save_dir)    #如果目录不存在，则生成该目录
    plt.savefig(os.path.join(save_dir,title))
    plt.close('all')

if "__name__" == "__main__":
    
    model_path = '/cptjack/totem/zhaofeiyan/DataSet/DataSet8-h5/ResNet50_1030.h5'
    model = load_model(model_path)
    svs_dir = "/cptjack/totem/Lung_animal_model/Test KCI/Right lobe"
    save_dir = "/cptjack/totem/zhaofeiyan/result/Right" 
    patch_size = 512
    level = 4
    svs_file = glob.glob(os.path.join(svs_dir, "*.ndpi"))
    svs_file = sorted(svs_file)
    
    for svs in tqdm(svs_file):
        if svs.split('.')[-1] == 'ndpi':
        
            slide = opsl.OpenSlide(svs)
            svs_name = os.path.basename(svs).split('.')[0] 
            thumbnail = slide.get_thumbnail(slide.level_dimensions[4]) 
            thum = np.asarray(thumbnail)
            img1 = get_mask(thum)
            out_img = get_out_img(model,svs,img1,patch_size,level)
            colormap(thumbnail, out_img, svs_name, save_dir)