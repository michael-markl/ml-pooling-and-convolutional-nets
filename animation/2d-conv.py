import cv2
import numpy as np

def img_to_tikz(image):
    print(r"\begin{tikzpicture}")

    print("\\matrix (mat) [table]")
    print("{")

    for i in range(image.shape[0]):
        line = ""
        for j in range(image.shape[1]):
            line += f"|[fill=white!{image[i][j]}!black] |"
            if j < image.shape[1] - 1:
                line += " & "
            else:
                line += "\\\\"
        print(line)
    print("};")
    print(r"\end{tikzpicture}")



image = cv2.imread('animation/4.png', 0)
#image = cv2.resize(image, (20, 20))
img_to_tikz(image)

kernel = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
]) / 4

#kernel = np.array([
#    [1, 1, 1],
#    [1, 1, 1],
#    [1, 1, 1]
#]) / 9

convolved = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_ISOLATED)
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.imshow('image',convolved)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_to_tikz(convolved)