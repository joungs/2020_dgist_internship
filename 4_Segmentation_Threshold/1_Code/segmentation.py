import cv2
import glob
import numpy as np
import random
from matplotlib import pyplot as plt


def image_cropping(image):
    b = image[:, :, 0]
    b1 = b.copy()
    hist0, bins = np.histogram(b, 256, [0, 256])
    g = image[:, :, 1]
    g1 = g.copy()
    hist1, bins = np.histogram(g, 256, [0, 256])
    r = image[:, :, 2]
    r1 = r.copy()
    hist2, bins = np.histogram(r, 256, [0, 256])
    # plt.hist(b.ravel(),256,[0,256]); plt.show()
    max = hist0.max()

    if max > 2500:
        edge = cv2.Laplacian(b, cv2.CV_64F)
        blur = cv2.GaussianBlur(edge, (9, 9), 5)
        blur = blur.astype(np.uint8)

        thresh1 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 2)
        img_contours1 = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
        img_contours1 = sorted(img_contours1, key=cv2.contourArea)

        for i in img_contours1:
            if cv2.contourArea(i) > 100000000:
                break
        mask1 = np.zeros(blur.shape[:2], np.uint8)
        cv2.drawContours(mask1, [i], -1, 255, -1)
        image = cv2.bitwise_and(image, image, mask=mask1)

    else:
        r1 = cv2.equalizeHist(r1)
        thresh1 = cv2.adaptiveThreshold(r1,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,2)
        img_contours1 = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
        img_contours1 = sorted(img_contours1, key=cv2.contourArea)

        for i in img_contours1:
            if cv2.contourArea(i) > 100000000:
                break
        mask1 = np.zeros(r1.shape[:2], np.uint8)
        cv2.drawContours(mask1, [i],-1, 255, -1)
        image = cv2.bitwise_and(image, image, mask=mask1)
    return image

def image_segmentation(imagee):
    b = imagee[:, :, 0]
    g = imagee[:, :, 1]
    r = imagee[:, :, 2]

    gray_img = cv2.medianBlur(g,9)
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    gray_img = cv2.filter2D(b, -1, kernel)

    kernel = np.ones((3,3), np.uint8)
    gray_img = cv2.erode(gray_img, kernel, iterations=1)
    gray_img = cv2.medianBlur(gray_img,3)

    asd = np.asarray(gray_img)
    max = asd.mean()
    image = gray_img.astype(np.uint8)
    ret3, th3 = cv2.threshold(image, max, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    gray_img = cv2.bilateralFilter(th3, 9,75,75)

    kernel = np.ones((2,2), np.uint8)
    image = cv2.erode(gray_img, kernel, iterations=1)
    return image
if __name__ == '__main__':

    path = '/Users/JunJoung/PycharmProjects/beef_segmentation/pp'   # your image path

    image_names = glob.glob('%s/*.jpg' % path)          # path & names of all images

    images = [cv2.imread(name, cv2.IMREAD_COLOR) for name in image_names]       # read all images
    images = [image_cropping(image) for image in images]                        # processing images
    images = [image_segmentation(imagee) for imagee in images]
    random.shuffle(images)          # shuffle images
    images = images[0:10]       # select 10 random images

    for index, image in enumerate(images):
        cv2.imshow('image%d' % index, image)
    cv2.waitKey(0)

