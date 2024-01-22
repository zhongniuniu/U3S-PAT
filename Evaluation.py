import os
import numpy as np
import math
from PIL import Image
from sklearn import metrics
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import numpy as np
import cv2
from skimage.metrics import mean_squared_error as compare_mse

def dice(s2, s1):
    row, col = s2.shape[0], s2.shape[1]
    d = []
    s = []
    for r in range(row - 10):
        for c in range(col - 10):
            if s1[r][c] == s2[r][c]:  # 计算图像像素交集
                s.append(s1[r][c])
    #                 print(s1[r][c])
    m1 = np.linalg.norm(s)
    m2 = np.linalg.norm(s1.flatten()) + np.linalg.norm(s2.flatten())
    d.append(2 * m1 / m2)
    return d

def normalization(x):
    return (x - x.min()) / (x.max() - x.min()) * 2 - 1


path1 = r'C:\Users\Admin\Desktop\ex\ex2-rate8-scan6\recon\output\DS\scan1/'  # 指定输出结果文件夹
path2 = r'C:\Users\Admin\Desktop\ex\ex2-rate8-scan6\recon\output\prior\scan1/'  # 指定原图文件夹
SSIM = 0
PSNR = 0
Dice = 0
num = 0

for i in range(1):
    # img_a = cv2.imread(path1 + str(i) + '.jpg')
    # img_b = cv2.imread(path2 + str(i) + '.png')
    img_a = cv2.imread(r'C:\Users\Admin\Desktop\ex\ex6\ls0.3\HGH324\Nd21\scan3-4\DS\ultra\HB.jpg')
    img_b = cv2.imread(r'C:\Users\Admin\Desktop\ex\ex6\ls0.3\HGH324\Nd21\scan3-4\output\0.25-0.25\HB.jpg')
    img_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
    img_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)
    img_a = normalization(img_a)
    img_b = normalization(img_b)
    # img_b = cv2.imread(path2 + str(i) + '.jpg')
    # img_b = cv2.resize(img_b, (288, 288))
    # img_a = np.array(img_a)
    # img_b = np.array(img_b)
    # img_a = normalization(img_a)
    # img_b = normalization(img_b)
    ssim = compare_ssim(img_b, img_a, multichannel=True)
    psnr = compare_psnr(img_b, img_a)
    print(dice(img_b, img_a))

    SSIM += ssim
    PSNR += psnr
    num += 1
    # print('SSIM:', ssim)
    # print('PSNR:', psnr)
print('SSIM:', SSIM / num)
print('PSNR:', PSNR / num)

