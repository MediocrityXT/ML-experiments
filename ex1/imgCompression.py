import numpy as np
import matplotlib.pyplot as pyplot
import imageio as img

def run():
    rawImg = img.imread('butterfly.bmp')
    height = rawImg.shape[0]
    width = rawImg.shape[1]
    b = np.empty((height, width))
    g = np.empty((height, width))
    r = np.empty((height, width))
    for i in range(height):
        for j in range(width):
            b[i][j] = rawImg[i][j][0]
            g[i][j] = rawImg[i][j][1]
            r[i][j] = rawImg[i][j][2]
    b_SVD = SVD(b, 5)
    g_SVD = SVD(g, 5)
    r_SVD = SVD(r, 5)
    printRGB_Img(r_SVD, g_SVD, b_SVD, height, width)

def SVD(pixels, k):
    U, S, V = np.linalg.svd(pixels)
    if k<1:
        #当k不合法时自适应寻找压缩比率大于某数的k（根据清晰度要求选择压缩比率下界）
        k= findK(len(U[:]),len(V[0]),0.3)
        print(k)
    singular = np.diag(S[:k])
    u = U[:, :k].reshape(len(U[:]), k)
    v = V[:k, :].reshape(k, len(V[0]))
    res = np.dot(np.dot(u,singular),v)
    res_uint = (res+0.5).astype('uint8')
    return res_uint

def findK(h,w,x):
    target = h*w*x
    s=0
    n=0
    while(s<target):
        n += 1
        s = h*n + n*n + n*w
    return n

def printRGB_Img(r, g, b, height, width):
    pixels = np.empty((height, width, 3), dtype='uint8')
    for i in range(height):
        for j in range(width):
            pixels[i][j][0] = b[i][j]
            pixels[i][j][1] = g[i][j]
            pixels[i][j][2] = r[i][j]
    pyplot.imshow(pixels)
    pyplot.show()


def printB_Img(b, height, width):
    B_Img = np.empty((height, width, 3), dtype='uint8')
    for i in range(height):
        for j in range(width):
            B_Img[i][j][0] = b[i][j]
            B_Img[i][j][1] = 0
            B_Img[i][j][2] = 0
    pyplot.imshow(B_Img)
    pyplot.show()


def printG_Img(g, height, width):
    G_Img = np.empty((height, width, 3), dtype='uint8')
    for i in range(height):
        for j in range(width):
            G_Img[i][j][0] = 0
            G_Img[i][j][1] = g[i][j]
            G_Img[i][j][2] = 0
    pyplot.imshow(G_Img)
    pyplot.show()


def printR_Img(r, height, width):
    R_Img = np.empty((height, width, 3), dtype='uint8')
    for i in range(height):
        for j in range(width):
            R_Img[i][j][0] = 0
            R_Img[i][j][1] = 0
            R_Img[i][j][2] = r[i][j]
    pyplot.imshow(R_Img)
    pyplot.show()
#
# def run2():
#     rawImg = img.imread('butterfly.bmp')
#     # pyplot.imshow(rawImg)
#     # 第一行 按行拼接
#     pixels = np.empty((106191, 3), dtype='uint8')
#     height = rawImg.shape[0]
#     width = rawImg.shape[1]
#     for i in range(height):
#         for j in range(width):
#             pixels[width * i + j] = rawImg[i][j]
#
#     printImg(SVD(pixels, 1), width, height)
#     printImg(SVD(pixels, 2), width, height)
#     printImg(SVD(pixels, 3), width, height)
#
# def printImg(pixels, width, height):
#     processedImg = np.empty([height, width, 3], dtype='uint8')
#     for i in range(height):
#         processedImg[i] = pixels[width * i:width * (i + 1)]
#     pyplot.imshow(processedImg)
#     pyplot.show()
#     return processedImg