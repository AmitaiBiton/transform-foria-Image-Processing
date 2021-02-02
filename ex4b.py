import numpy as np
import cv2
from matplotlib.pyplot import figure
figure(num=None, figsize=(16, 16), dpi=80, facecolor='w', edgecolor='k')
from matplotlib import pyplot as plt

def create_filters(size):

    f1  = np.ones((size))
    f2 = np.ones((size))
    f3 = np.zeros((size))
    f4 = np.zeros((size))
    for i in range(40):
        f1[: , int(size[0]/2)-20+i] = 0
        f2[int(size[0]/2)-20+i,:]=0
        f3[:, int(size[0] / 2) - 20 + i] = 1
        f4[int(size[0] / 2) - 20 + i, :] = 1
    f5 = f1+f2
    f5[f5>0] =1
    f6  = np.zeros((size))
    f6[f5==1]=0
    f6[f5==0] =1
    f7  = f3+f4
    f7[f7>0]=1
    f8 = np.zeros((size))
    f8[f7==1]=0
    f8[f7==0]=1
    return f1, f2,f3,f4,f5,f6,f7,f8


def FFT(image , f1):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    final = fshift*f1

    final = np.fft.ifftshift(final)
    final = np.fft.ifft2(final)
    final = abs(final)
    final  = final.astype(np.uint64)
    return final

if  __name__ == "__main__":
    image = cv2.imread('pic.png' , 0)
    f1, f2,f3,f4,f5,f6,f7,f8 =   create_filters(image.shape)
    f1_image  = FFT(image ,f1)
    f2_image  = FFT(image ,f2)
    f3_image  = FFT(image ,f3)
    f4_image  = FFT(image ,f4)
    f5_image  = FFT(image ,f5)
    f6_image  = FFT(image ,f6)
    f7_image  = FFT(image ,f7)
    f8_image  = FFT(image ,f8)



    plt.subplot(2, 4, 1), plt.imshow(f1_image, cmap="gray"), plt.title('f1_image')
    plt.xticks([]), plt.yticks([])

    plt.subplot(2, 4, 2), plt.imshow(f2_image, cmap="gray"), plt.title('f2_image')
    plt.xticks([]), plt.yticks([])

    plt.subplot(2, 4, 3), plt.imshow(f3_image, cmap="gray"), plt.title('f3_image')
    plt.xticks([]), plt.yticks([])
    plt.subplot(2, 4, 4), plt.imshow(f4_image, cmap="gray"), plt.title('f4_image')
    plt.xticks([]), plt.yticks([])
    plt.subplot(2, 4, 5), plt.imshow(f5_image, cmap="gray"), plt.title('f5_image')
    plt.xticks([]), plt.yticks([])
    plt.subplot(2, 4, 6), plt.imshow(f6_image, cmap="gray"), plt.title('f6_image')
    plt.xticks([]), plt.yticks([])
    plt.subplot(2, 4, 7), plt.imshow(f7_image, cmap="gray"), plt.title('f7_image')
    plt.xticks([]), plt.yticks([])
    plt.subplot(2, 4, 8), plt.imshow(f8_image, cmap="gray"), plt.title('f8_image')
    plt.xticks([]), plt.yticks([])

    plt.show()