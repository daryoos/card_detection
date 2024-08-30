import cv2 as cv


def automaticBinarization(image):
    # Calculate histogram
    histogram = cv.calcHist([image], [0], None, [256], [0, 256]).flatten()

    i_min, i_max = 0, 0
    for i in range(255, -1, -1):
        if histogram[i] > 0:
            i_max = i
            break
    for i in range(256):
        if histogram[i] > 0:
            i_min = i
            break

    threshold1 = 0.0
    threshold2 = float(i_min + i_max) / 2
    error = 0.1

    while not ((threshold2 - threshold1) < error):
        threshold1 = threshold2
        mean1 = mean2 = 0.0
        n1 = n2 = 0

        for i in range(i_min, int(threshold1)):
            mean1 += histogram[i] * i
            n1 += histogram[i]

        for i in range(int(threshold1) + 1, i_max):
            mean2 += histogram[i] * i
            n2 += histogram[i]

        threshold2 = ((mean1 / n1 if n1 != 0 else 0) + (mean2 / n2 if n2 != 0 else 0)) / 2

    _, binarized = cv.threshold(image, threshold2, 255, cv.THRESH_BINARY)

    return binarized


def findBiggestContourAndResizeByBoundingRectangle(image, width, height):
    contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    sized = []
    if len(contours) != 0:
        contours = sorted(contours, key=cv.contourArea, reverse=True)

        contour = contours[1]

        x, y, w, h = cv.boundingRect(contour)

        roi = image[y:y + h, x:x + w]
        sized = cv.resize(roi, (width, height), 0, 0)

    return sized


def preprocessImage(image):
    # Convert to gray scale and apply a filter
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)

    binarized = automaticBinarization(blur)
    return binarized
