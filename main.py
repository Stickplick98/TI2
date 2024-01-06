"""
import cv2 as cv
import numpy as np

def geodesic_reconstruction(marker, mask, kernel_size=2):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Dilatation répétée du marqueur jusqu'à la convergence
    marker_dilated = cv.dilate(marker, kernel, iterations=10)
    marker_dilated = cv.erode(marker_dilated, kernel, iterations=10)

    return marker_dilated


blue = np.uint8([[[0, 255, 0]]])
hsv_blue = cv.cvtColor(blue, cv.COLOR_BGR2HSV)
print(hsv_blue)

red = np.uint8([[[0, 0, 255]]])
hsv_red = cv.cvtColor(red, cv.COLOR_BGR2HSV)
print(hsv_red)

img = cv.imread("C:\\TraitementImages3eme\\Images\\petitsPois.png", cv.IMREAD_COLOR)
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow("Image", img)

lower_blue = np.array([100, 150, 150])
upper_blue = np.array([130, 255, 255])

lower_red = np.array([0, 150, 150])
upper_red = np.array([30, 255, 255])

markerb, markerg, markerr = cv.split(img)

mask = cv.inRange(img_hsv, lower_blue, upper_blue)
result = geodesic_reconstruction(markerb, mask)

hauteur, largeur, _ = img.shape
image_blanche = np.ones((hauteur, largeur, 1), dtype=np.uint8) * 255
image_fusionnee = cv.merge([result, result, image_blanche])

mask = cv.inRange(img_hsv, lower_red, upper_red)

cv.imshow("Resultat masque", mask)
result2 = geodesic_reconstruction(markerr, mask)
image_fusionnee2 = cv.merge([image_blanche, result2, result2])

cv.imshow("Resultat Rouge", image_fusionnee)
cv.imshow("Resultat Bleu", image_fusionnee2)

cv.waitKey(0)
cv.destroyAllWindows()

"""

import cv2 as cv
import numpy as np


def geodesic_reconstruction(marker, kernel_size=2):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Dilatation répétée du marqueur jusqu'à la convergence
    marker_dilated = cv.dilate(marker, kernel, iterations=10)
    marker_dilated = cv.erode(marker_dilated, kernel, iterations=10)

    return marker_dilated


img = cv.imread("C:\\TraitementImages3eme\\Images\\petitsPois.png", cv.IMREAD_COLOR)
cv.imshow("Image", img)

markerb, markerg, markerr = cv.split(img)

result = geodesic_reconstruction(markerb)

hauteur, largeur, _ = img.shape
image_blanche = np.ones((hauteur, largeur, 1), dtype=np.uint8) * 255
image_fusionnee = cv.merge([result, result, image_blanche])

result2 = geodesic_reconstruction(markerr)
image_fusionnee2 = cv.merge([image_blanche, result2, result2])

cv.imshow("Resultat ", result2)
cv.imshow("Resultat Rouge", image_fusionnee)
cv.imshow("Resultat Bleu", image_fusionnee2)

cv.waitKey(0)
cv.destroyAllWindows()
