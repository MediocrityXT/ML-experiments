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
    b_U, b_S, b_V = SVD(b, 5)
    g_U, g_S, g_V = SVD(g, 5)
    r_U, r_S, r_V = SVD(r, 5)
    b_SVD = SVDtoPixels(b_U, b_S, b_V )
    g_SVD = SVDtoPixels(g_U, g_S, g_V)
    r_SVD = SVDtoPixels(r_U, r_S, r_V)
    pixels=printRGB_Img(r_SVD, g_SVD, b_SVD, height, width)
    pyplot.subplot(1, 2, 2)
    pyplot.imshow(pixels)
    pyplot.show()

def SVD(pixels, k):
    U, S, V = np.linalg.svd(pixels)
    u = U[:, :k].reshape(len(U[:]), k)
    v = V[:k, :].reshape(k, len(V[0]))
    return u, S[:k], v

def SVDtoPixels(U,S,V):
    singular = np.diag(S)
    res = np.dot(np.dot(U, singular), V)
    res_uint = (res + 0.5).astype('uint8')
    return res_uint

def printRGB_Img(r, g, b, height, width):
    pixels = np.empty((height, width, 3), dtype='uint8')
    for i in range(height):
        for j in range(width):
            pixels[i][j][0] = b[i][j]
            pixels[i][j][1] = g[i][j]
            pixels[i][j][2] = r[i][j]
    return pixels