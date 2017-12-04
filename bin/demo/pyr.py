import cv2

imgs = [cv2.imread("<YOUR_IMAGE_PATH>")]
for _ in range(3):
    r = cv2.pyrDown(imgs[-1])
    imgs.append(r)
for (i, img) in enumerate(imgs):
    cv2.imshow("view_%d" % i, img)

cv2.waitKey()
