
"""
Created on Mon Aug 26 09:47:24 2019

@author: zhaofy

该脚本用来读取xml标注内容，并根据标注生成ndpi文件中ignore区域的mask图

"""

import xml.etree.ElementTree as ET
import openslide as opsl
from PIL import ImageDraw
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np


def dist(a,b):
    return round(abs(a[0]-b[0]) + abs(a[1]-b[1]))
    
def read_xml(xml_file):

    """
    从xml文件中读取坐标点（x,y）

    """
    tree = ET.parse(xml_file)
    region_list = []
   
    for region in tree.findall('.//Annotation'):
        if region.get('LineColor') =='65535':    
            for region in tree.findall('.//Annotation/Regions/Region'):
                vertex_list = []
                if region.attrib.get('Type')=='0':
                    for vertex in region.findall('.//Vertices/Vertex'):
                    # parse the 'X' and 'Y' for the vertex
                        vertex_list.append(vertex.attrib)
                    region_list.append(vertex_list)

    return region_list


def region_to_image(svs,region_list, level_downsample_number):

    """
    
    根据读取到的坐标获取ignore区域的mask图，并在ndpi大图的四级缩略图下显示出来
    
    svs：ndpi文件路径
    region_list：读取到的坐标列表，[(x1,y1),(x2,y2),(x3,y3),...]
    level_downsample_number：大图的级数
    
    """
    regions_list = []
    slide = opsl.OpenSlide(svs)
    level_downsample = slide.level_downsamples[level_downsample_number]
    level_dimension = slide.level_dimensions[level_downsample_number]
    thumbnail = Image.new(mode = "1",size = level_dimension)
    dr = ImageDraw.Draw(thumbnail)
    
    for _, region in enumerate(region_list):
        point_list = []
        for _, point in enumerate(region):
            X = int(float(point['X']) / level_downsample)
            Y = int(float(point['Y']) / level_downsample)
            point_list.append((X, Y))
        
        regions_list.append(point_list)
        #print(regions_list)
        
    pin_jie_flag = [] #存储已经被拼接过的标注坐标列表序号                  
    single_list = [] #存储新标注坐标列表的列表 
    
    for j,p_list in enumerate(regions_list):
        if dist(p_list[0], p_list[-1]) < 100 and j not in pin_jie_flag:
        #如果收尾坐标距离相差在150范围内(曼哈顿距离)，且未成被拼接过，直接认为这个组坐标无须拼接，存储起来
            single_list.append(p_list)                
        elif dist(p_list[0], p_list[-1]) > 100 and j not in pin_jie_flag:
        #如果收尾坐标距离相差在150范围外(曼哈顿距离)，且未成被拼接过，说明这组坐标是残缺非闭合的，需要对其余标注坐标进行新一轮的循环判断
            for j_2,p_list_2 in enumerate(regions_list):
                while j_2 != j and dist(p_list[-1],p_list_2[0]) < 100 and j_2 not in pin_jie_flag:
                    p_list = p_list + p_list_2.copy()
                    # 当这组非闭合的尾坐标和其他组坐标的首坐标接近到一定范围时(距离是150内),就让当前的非闭合的坐标列表和该组坐标列表相加
                    pin_jie_flag.append(j_2)
                    # 处理完毕之后，将该组坐标的序号增加到已拼接坐标的列表中，确保后续循环不会再判断这个列表
            single_list.append(p_list)
            #print(single_list)


###      如果大图中有多个ignore区域，比如Right里面的4-R.ndpi，就用下面这个方法依此读取出ignore的坐标

#        thumbnail = Image.new(mode = "1",size = level_dimension)        
#        dr = ImageDraw.Draw(thumbnail)                   
#        dr.polygon(single_list[3], fill="#ffffff")
#        
#        photo = np.array(thumbnail).astype(np.uint8)
#       
##        cv2.imwrite("/cptjack/totem/zhaofeiyan/image/region/photo2/32.jpg",photo*255)   
##        plt.imsave('/cptjack/totem/zhaofeiyan/image/region/photo1/3.jpg',photo)   
##        np.save("/cptjack/totem/zhaofeiyan/image/region/numpy2/32-R.npy",photo)
#        plt.imshow(photo)  
        
        
###      如果大图中只有一个ignore区域，就用下面这个方法读取ignore的坐标
   
    for points in regions_list:
        dr.polygon(points, fill="#ffffff")

    #由于医生的标注除了出现不连续(非闭合)的情况外，还存在多余勾画的情况，对这种情况暂时没有完整的思路予以接近，先用
    # opencv中的开闭操作组合来进行修补
    kernel = np.ones((20,20),np.uint8)
    filter_matrix = np.array(thumbnail).astype(np.uint8)
    filter_matrix = cv2.morphologyEx(filter_matrix, cv2.MORPH_OPEN, kernel)
    filter_matrix = cv2.morphologyEx(filter_matrix, cv2.MORPH_CLOSE, kernel)

#    np.save("/cptjack/totem/zhaofeiyan/image/region/numpy/4-11.npy",filter_matrix)
#    cv2.imwrite("/cptjack/totem/zhaofeiyan/image/region/photo1/4-11.jpg",filter_matrix)
    
    plt.imshow(filter_matrix)
     
    return filter_matrix
    
if __name__ == "__main__" :
    
    xml_file = '/cptjack/totem/Lung_animal_model/Test KCI/Right lobe/4-R.xml'
    ndpi_file = '/cptjack/totem/Lung_animal_model/Test KCI/Right lobe/4-R.ndpi'

    region_list = read_xml(xml_file)
    region_to_image(ndpi_file, region_list, 4)
