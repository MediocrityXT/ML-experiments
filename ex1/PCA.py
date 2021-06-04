import numpy as np
import matplotlib.pyplot as pyplot
import imageio as img


def run():
    rawImg = img.imread('butterfly.bmp')
    pyplot.subplot(1, 2, 1)
    pyplot.imshow(rawImg)
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
    bPCA, b_u = PCA(b, 5)
    gPCA, g_u = PCA(g, 5)
    rPCA, r_u = PCA(r, 5)
    pixels = printPCA(bPCA, b_u, gPCA, g_u, rPCA, r_u, height, width)
    pyplot.subplot(1, 2, 2)
    pyplot.imshow(pixels)
    pyplot.show()


def PCA(data, k):
    # 对于三维数据，计算协方差矩阵
    c = np.cov(data)
    e_vals, e_vecs = np.linalg.eig(c)
    sorted_indices = np.argsort(e_vals)
    e = e_vals[sorted_indices[:-k - 1:-1]]
    # 在从小到大排序的索引中倒着找k个
    v = e_vecs[:, sorted_indices[:-k - 1:-1]]
    vT = np.transpose(v)
    res = np.dot(vT, data)
    return res, v


def printPCA(bPCA, b_u, gPCA, g_u, rPCA, r_u, height, width):
    bdata = np.dot(b_u, bPCA)
    gdata = np.dot(g_u, gPCA)
    rdata = np.dot(r_u, rPCA)
    pixels = np.empty([height, width, 3], dtype='uint8')
    n = 0
    for i in range(height):
        for j in range(width):
            pixels[i][j][0] = cut(bdata[i][j])
            pixels[i][j][1] = cut(gdata[i][j])
            pixels[i][j][2] = cut(rdata[i][j])
            n += 1
    return pixels


def cut(n):
    if n < 0:
        return 0
    elif n > 255:
        return 255
    else:
        return int(n + 0.5)
