import scipy.fftpack as ftp
import cv2
import numpy as np
import math
from util import *


def convert_block_matrix(mtrx, unit):
    rows = int(math.ceil(float(mtrx.shape[0])/float(unit)))
    columns  = int(math.ceil(float(mtrx.shape[1])/float(unit)))

    divided_mtrx = [ [ [[]] for i in range(columns) ] for j in range(rows) ]
    for i in range(0, rows):
        for j in range(0, columns):
            divided_mtrx[i][j] = mtrx[i*unit:(i+1)*unit,j*unit:(j+1)*unit]

    return divided_mtrx


def discrete_cosine_transform(divided_mtrx):
    rows = len(divided_mtrx)
    columns = len(divided_mtrx[0])
    
    dct_mtrx = [ [ [[]] for i in range(rows) ] for j in range(columns) ]

    for i in range(0, rows):
        for j in range(0, columns):
            dct_mtrx[i][j] = ftp.dct(divided_mtrx[i][j])

    return dct_mtrx


def inverse_discrete_cosine_transform(dct_mtrx):
    rows = len(dct_mtrx)
    columns = len(dct_mtrx[0])

    divided_mtrx =[ [ [[]] for i in range(rows) ] for j in range(columns) ]
    for i in range(0, rows):
        for j in range(0, columns):
            divided_mtrx[i][j] = ftp.dct(x=dct_mtrx[i][j], type=3)
    
    return divided_mtrx


def convert_original_matrix(divided_mtrx, unit):
    columns = unit*len(divided_mtrx)
    rows = unit*len(divided_mtrx[0])

    mtrx = [[ 0 for i in range(columns) ] for j in range(rows)]
    for i in range(0, rows):
        for j in range(0, columns):
            mtrx[i][j] = divided_mtrx[int(math.floor(float(i)/float(unit)))][int(math.floor(float(j)/float(unit)))][i % unit][j % unit]
    
    return np.array(mtrx)


def PSNR(a, b):
    MSE_SUM = 0.0
    MAX = 0.0

    for i in range(len(a)):
        for j in range(len(a[i])):
            MSE_SUM += (a[i][j] - b[i][j]) ** 2

            if MAX < a[i][j]:
                MAX = a[i][j]

    MSE = float(MSE_SUM) / float(len(a)*len(a[0]))

    return 10*math.log10(float(MAX)**2 / MSE)


if __name__ == '__main__':
    im = cv2.imread('LENNA.png', 0)
    
    print im
    bm = convert_block_matrix(im, 8)
    dctm = discrete_cosine_transform(bm)

    tri_matrix = triangle(8)

    for i in range(len(dctm)):
        for j in range(len(dctm[i])):
            dctm[i][j] = tri_matrix * dctm[i][j]

    idctm = inverse_discrete_cosine_transform(dctm)
    comp_image = convert_original_matrix(idctm, 8)*(2.0/32.0)

    print comp_image

    print PSNR(comp_image, im)
