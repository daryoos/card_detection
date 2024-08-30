import cv2 as cv
import numpy as np
import ImageProcessing
import Cards


def main():
    image = cv.imread('C:\\Users\\dariu\\PycharmProjects\\CardDetection\\images\\input\\testImg1.bmp')

    cv.imshow("image", image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    cards = Cards.getAllCardFromImage(image)
    cv.destroyAllWindows()

    for card in cards:
        print(card.rank + " of " + card.suit)


if __name__ == "__main__":
    main()
