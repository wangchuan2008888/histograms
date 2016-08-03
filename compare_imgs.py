import sys

from scipy.misc import imread
from scipy.linalg import norm
from scipy import sum, average

'''
From http://stackoverflow.com/questions/189943/how-can-i-quantify-difference-between-two-images
'''

def main():
    file1, file2 = sys.argv[1:1+2]
    # read images as 3D arrays
    img1 = imread(file1).astype(float)
    img2 = imread(file2).astype(float)
    # compare
    n_0 = compare_images(img1, img2)
    #print "Manhattan norm:", n_m, "/ per pixel:", n_m/img1.size
    print "Zero norm:", n_0, "/ per pixel:", n_0*1.0/img1.size

def compare_images(img1, img2):
    # normalize to compensate for exposure difference, this may be unnecessary
    # consider disabling it
    img1 = normalize(img1)
    img2 = normalize(img2)
    # calculate the difference and its norms
    diff = img1 - img2  # elementwise for scipy arrays
    #m_norm = sum(abs(diff))  # Manhattan norm
    z_norm = norm(diff.ravel(), 0)  # Zero norm
    return z_norm#(m_norm, z_norm)

def normalize(arr):
    rng = arr.max()-arr.min()
    amin = arr.min()
    return (arr-amin)*255/rng

if __name__ == "__main__":
    main()
