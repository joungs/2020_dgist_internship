import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('/Users/JunJoung/PycharmProjects/beef_segmentation/pp/a2.jpg')


b = image[:, :, 0]
b1 = b.copy()
hist0, bins = np.histogram(b, 256, [0, 256])
g = image[:, :, 1]
g1 = g.copy()
hist1, bins = np.histogram(g, 256, [0, 256])
r = image[:, :, 2]
r1 = r.copy()
hist2, bins = np.histogram(r, 256, [0, 256])
# plt.hist(r.ravel(),256,[0,256]); plt.show()
max = hist2.max()

if max > 5000:
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

    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]

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


    cv2.imshow('color', gray_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
    # image = cv2.GaussianBlur(image, (1,1),0)

    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]

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
    gray_img = cv2.erode(gray_img, kernel, iterations=1)


    cv2.imshow('color', gray_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def image_segmentation(imagee):
    gray_img = (imagee[:, :, 0] + imagee[:, :, 1] + imagee[:, :, 2]) /3
    # gray_img = gray_img.astype(np.uint8)
    asd = np.asarray(gray_img)
    max = asd.mean()
    blur = cv2.GaussianBlur(gray_img, (3, 3), 3)
    blur = blur.astype(np.uint8)
    ret3, th3 = cv2.threshold(blur, max, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image = th3
    return image

# image = img[:, :, 2]
# # image = image.astype(np.uint8)
# # r = image[:, :, 2]
# r1 = cv2.equalizeHist(image)
# # r1 = cv2.GaussianBlur(r1, (9, 9), 5)
# thresh1 = cv2.adaptiveThreshold(r1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)
# img_contours1 = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
# img_contours1 = sorted(img_contours1, key=cv2.contourArea)
#
# for i in img_contours1:
#     if cv2.contourArea(i) > 100000000:
#         break
# mask1 = np.zeros(r1.shape[:2], np.uint8)
# cv2.drawContours(mask1, [i], -1, 255, -1)
# image = cv2.bitwise_and(img, img, mask=mask1)
#
#
# gray_img = cv2.Laplacian(image,cv2.CV_16UC3)
# asd = np.asarray(gray_img)
# max = asd.mean()
# blur = cv2.GaussianBlur(gray_img, (9, 9), 5)
# # blur = blur.astype(np.uint8)
# ret2,thresh1 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# cv2.imshow('image', thresh1)
# cv2.waitKey(0)

# b = img[:,:,0]
# hist0, bins = np.histogram(b, 256,[0,256])
# g = img[:,:,1]
# hist1, bins = np.histogram(g, 256,[0,256])
# r = img[:,:,2]
# hist2, bins = np.histogram(r, 256,[0,256])
# # plt.hist(b.ravel(),256,[0,256]); plt.show()
# max = np.argmax(hist0) - 1
#
# edge = cv2.Laplacian(b,cv2.CV_64F)
# blur = cv2.GaussianBlur(edge, (9,9),5)
# blur = blur.astype(np.uint8)
#
# thresh1 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 2)
# img_contours1 = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
# img_contours1 = sorted(img_contours1, key=cv2.contourArea)
#
# for i in img_contours1:
#     if cv2.contourArea(i) > 100000000:
#         break
# mask1 = np.zeros(blur.shape[:2], np.uint8)
# cv2.drawContours(mask1, [i], -1, 255, -1)
# new_img1 = cv2.bitwise_and(b, b, mask=mask1)
#
#
# edge2 = cv2.Laplacian(g,cv2.CV_64F)
# blur2 = cv2.GaussianBlur(edge2, (9,9),5)
# blur2 = blur2.astype(np.uint8)
#
# thresh2 = cv2.adaptiveThreshold(blur2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 2)
# img_contours2 = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
# img_contours2 = sorted(img_contours2, key=cv2.contourArea)
#
# for i in img_contours2:
#     if cv2.contourArea(i) > 100000000:
#         break
# mask2 = np.zeros(blur2.shape[:2], np.uint8)
# cv2.drawContours(mask2, [i], -1, 255, -1)
# new_img2 = cv2.bitwise_and(g, g, mask=mask2)
#
#
# edge3 = cv2.Laplacian(r,cv2.CV_64F)
# blur3 = cv2.GaussianBlur(edge3, (9,9),5)
# blur3 = blur3.astype(np.uint8)
#
# thresh3 = cv2.adaptiveThreshold(blur3, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 2)
# img_contours3 = cv2.findContours(thresh3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
# img_contours3 = sorted(img_contours3, key=cv2.contourArea)
#
# for i in img_contours3:
#     if cv2.contourArea(i) > 100000000:
#         break
# mask3 = np.zeros(blur3.shape[:2], np.uint8)
# cv2.drawContours(mask3, [i], -1, 255, -1)
# new_img3 = cv2.bitwise_and(r, r, mask=mask3)
#
# new = cv2.merge((new_img1, new_img2, new_img3))
# cv2.imshow('w', new)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# hist, bins = np.histogram(im, 256,[0,256])
# max = np.argmax(hist) - 1
#
#
# image = cv2.GaussianBlur(im, (5,5,),3 )
# image = cv2.Canny(image, 80,120)
# cv2.imshow('img', image)
# cv2.waitKey(0)


# hist, bins = np.histogram(img, 256,[0,256])
# max = hist.max()
# r1 = cv2.equalizeHist(r1)
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# r1 = clahe.apply(r1)
#
# blur = cv2.GaussianBlur(r1,(5,5),0)
# r1 = cv2.addWeighted(blur,1.5,r1,-0.5,0)
#
# filter = np.array([[1, 1, 1], [1, -7, 1], [1, 1, 1]])
# r1=cv2.filter2D(r1,-1,filter)
#
# print(hist)
# if max > 1500:
#     thresh1 = cv2.adaptiveThreshold(r1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 2)
#     img_contours1 = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
#     img_contours1 = sorted(img_contours1, key=cv2.contourArea)
#
#     for i in img_contours1:
#         if cv2.contourArea(i) > 100000000:
#             break
#     mask1 = np.zeros(r1.shape[:2], np.uint8)
#     cv2.drawContours(mask1, [i], -1, 255, -1)
#     new_img1 = cv2.bitwise_and(img, img, mask=mask1)
#     cv2.imshow('w', new_img1)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     thresh1 = cv2.adaptiveThreshold(r1,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,2)
#     img_contours1 = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
#     img_contours1 = sorted(img_contours1, key=cv2.contourArea)
#
#     for i in img_contours1:
#         if cv2.contourArea(i) > 100000000:
#             break
#     mask1 = np.zeros(r1.shape[:2], np.uint8)
#     cv2.drawContours(mask1, [i],-1, 255, -1)
#     new_img1 = cv2.bitwise_and(img, img, mask=mask1)
#     cv2.imshow('w', new_img1)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# gray_img = cv2.cvtColor(new_img1, cv2.COLOR_BGR2GRAY)
#
#
# asd = np.asarray(gray_img)
# asd = asd.astype(np.float32)
# mean = asd.mean()
# print(mean)
#
# # hist, bins = np.histogram(img.flatten(), 256,[0,256])
# #
# # cdf = hist.cumsum()
# # cdf_m = np.ma.masked_equal(cdf,0)
# # cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
# # cdf = np.ma.filled(cdf_m,0).astype('uint8')
# # img2 = cdf[img]
#
#
# hist, bins = np.histogram(img, 256, [0, 256])
# max = np.argmax(hist) - 1
#
# ret, binary = cv2.threshold(gray_img, mean, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# cv2.imshow('bin', binary)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# plt.hist(gray_img.ravel(),256,[0,256]); plt.show()