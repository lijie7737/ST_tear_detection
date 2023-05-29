from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image
from tqdm import tqdm
import torchvision
import numpy as np
import torch
import torch.nn as nn
from piqa import ssim
import piqa
import pandas as pd
import numpy as np
from scipy.stats import shapiro,ttest_rel,ranksums
import shutil
import SimpleITK as sitk
import math

def Create_Folder(path_list):
    for Folder_path in path_list:
        if os.path.exists(Folder_path):
          shutil.rmtree(Folder_path)
        os.makedirs(Folder_path)# Create folder to store the .nii files
        print('Created Folder:', Folder_path)


def resampleImage_img_2D(image, targetSpacing, resamplemethod=sitk.sitkLinear):
    """
    将体数据重采样的指定的spacing大小
    paras:
    image:sitk读取的image信息,这里是体数据
    targetSpacing:指定的spacing,例如[1,1,1]
    resamplemethod:插值类型
    return:重采样后的数据
    """
    targetsize = [0, 0, 0]
    #读取原始数据的size和spacing信息
    ori_size = image.GetSize()
    ori_spacing = image.GetSpacing()
    transform = sitk.Transform()
    transform.SetIdentity()
    #计算改变spacing后的size，用物理尺寸/体素的大小
    targetsize[0] = round(ori_size[0] * ori_spacing[0] / targetSpacing[0])
    targetsize[1] = round(ori_size[1] * ori_spacing[1] / targetSpacing[1])
    targetsize[2] = round(ori_size[2])

    targetSpacing.append(ori_spacing[2])

    new_spacing = targetSpacing

    #设定重采样的一些参数
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetSize(targetsize)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(resamplemethod)
    if resamplemethod == sitk.sitkNearestNeighbor:
        #mask用最近邻插值，保存为uint8
        resampler.SetOutputPixelType(sitk.sitkUInt8)
    else:
    	#体数据用线性插值，保存为float32
        resampler.SetOutputPixelType(sitk.sitkFloat32)
    newImage = resampler.Execute(image)
    return newImage



def RescaleIntensity(img):
    """
    args:
    img:img
    """
    rescalFilt = sitk.RescaleIntensityImageFilter() 
    rescalFilt.SetOutputMaximum(255)
    rescalFilt.SetOutputMinimum(0)
    Rescale_img = rescalFilt.Execute(img)
    return Rescale_img


def get_max_roi_3d(label_img,label_valule):
    # 读取标签图像
    label = label_img

    # 确定要选择的标签值
    label_value = label_valule

    # 在标签中搜索选择的标签值
    label_array = sitk.GetArrayFromImage(label)
    selected_label_array = np.zeros_like(label_array)
    selected_label_array[label_array == label_value] = 1

    # 找到选择的标签的最大截面
    z_indices = np.nonzero(np.sum(selected_label_array, axis=(1, 2)))[0]
    z_min = np.min(z_indices)
    z_max = np.max(z_indices)

    y_indices = np.nonzero(np.sum(selected_label_array, axis=(0, 2)))[0]
    y_min = np.min(y_indices)
    y_max = np.max(y_indices)

    x_indices = np.nonzero(np.sum(selected_label_array, axis=(0, 1)))[0]
    x_min = np.min(x_indices)
    x_max = np.max(x_indices)
    return x_min,x_max,y_min,y_max,z_min,z_max


def reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result