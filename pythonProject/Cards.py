import cv2 as cv
import numpy as np
import ImageProcessing

CARD_MIN_AREA = 10000

CARD_WIDTH = 200
CARD_HEIGHT = 300

CORNER_WIDTH = 32
CORNER_HEIGHT = 84

RANK_WIDTH = 70
RANK_HEIGHT = 125

SUIT_WIDTH = 70
SUIT_HEIGHT = 100

ranks = np.array(["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"])
suits = np.array(["Hearts", "Diamonds", "Clubs", "Spades"])


class Card:
    def __init__(self):
        self.rank = ""
        self.suit = ""


def getCardContours(image):
    # Find contours
    contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return [], []

    # Sorting the lists
    index_sort = sorted(range(len(contours)), key=lambda i: cv.contourArea(contours[i]), reverse=True)

    sorted_contours = [contours[i] for i in index_sort]
    sorted_hierarchy = [hierarchy[0][i] for i in index_sort]

    # card_contours = np.zeros(len(sorted_contours), dtype=int)
    card_contours = []

    # Detect which of the contours are cards
    for i, contour in enumerate(sorted_contours):
        area = cv.contourArea(contour)

        # Check if contour area is bigger than our threshold
        if area > CARD_MIN_AREA:
            # Approximate contour
            perimeter = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.01 * perimeter, True)

            # Check if contour is a quadrilateral and has no parent
            if len(approx) == 4 and sorted_hierarchy[i][3] == -1:
                card_contours.append(contour)

    # contour_image = cv.cvtColor(src, cv.COLOR_GRAY2BGR)
    # cv.drawContours(contour_image, card_contours, -1, (0, 255, 0), 2)

    # cv.imshow('Contours', contour_image)
    # cv.waitKey(0)

    return card_contours


def transformCard(image, corner_points, card_width, card_height):
    corners_order = np.zeros((4, 2), dtype="float32")

    if corner_points.shape == (4, 1, 2):
        corner_points = corner_points.reshape(4, 2)

    # Calculate the sum and the diff of the coordinates
    sum_points = np.sum(corner_points, axis=1)
    diff_points = np.diff(corner_points, axis=1).flatten()

    # Get top left and bottom right corners
    # Top left has the smallest sum of x and y
    top_left = corner_points[np.argmin(sum_points)]
    # Bottom right has the largest sum of x and y
    bottom_right = corner_points[np.argmax(sum_points)]

    # Find top right and bottom left corners
    # Top right has the smallest difference between y and x
    top_right = corner_points[np.argmin(diff_points)]
    # Bottom left has the largest difference between y and x
    bottom_left = corner_points[np.argmax(diff_points)]

    # Calculate width and height
    width = np.linalg.norm(top_right - top_left)
    height = np.linalg.norm(bottom_left - top_left)

    # Determine if the card is vertically, horizontally, or diagonally oriented
    # Card is vertically oriented
    if width < height:
        # Correct order
        if top_left[1] < bottom_left[1]:
            corners_order = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
        # Swap top and bottom if order is incorrect
        else:
            corners_order = np.array([bottom_left, bottom_right, top_right, top_left], dtype="float32")
    else:
        # Card is horizontally oriented
        if width > height:
            # Correct order for horizontally oriented card
            if top_left[0] < bottom_right[0]:
                corners_order = np.array([top_left, bottom_left, bottom_right, top_right], dtype="float32")
            # Swap left and right if order is incorrect
            else:
                corners_order = np.array([top_right, bottom_right, bottom_left, top_left], dtype="float32")
        # Card is diagonally oriented
        else:
            # Determine tilt by checking the relative y coordinates of top right and bottom left
            # Tilted to the left
            if top_right[1] < bottom_left[1]:
                corners_order = np.array([top_right, top_left, bottom_left, bottom_right], dtype="float32")
            # Tilted to the right
            else:
                corners_order = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")

    dst_points = np.array([[0, 0], [CARD_WIDTH - 1, 0], [CARD_WIDTH - 1, CARD_HEIGHT - 1], [0, CARD_HEIGHT - 1]],
                          np.float32)
    M = cv.getPerspectiveTransform(corners_order, dst_points)
    warped = cv.warpPerspective(image, M, (CARD_WIDTH, CARD_HEIGHT))
    warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)

    return warped


def extractCard(image, card_contour):
    # Get corner points
    perimeter = cv.arcLength(card_contour, True)
    approx = cv.approxPolyDP(card_contour, 0.01 * perimeter, True)
    corner_points = np.float32(approx)

    # Get bounding Rectangle
    x, y, w, h = cv.boundingRect(card_contour)

    # Get the center points of the card
    average = np.sum(corner_points, axis=0) / len(corner_points)
    center_x = int(average[0][0])
    center_y = int(average[0][1])

    warp = transformCard(image, corner_points, w, h)

    return warp


def getCorner(warp):
    # Getting corner of the card
    corner = warp[0:CORNER_HEIGHT, 0:CORNER_WIDTH]
    # Zoom 4x on corner
    corner_zoom = cv.resize(corner, (0, 0), fx=4, fy=4)

    # cv.imshow('Corner Zoom', corner_zoom)
    # cv.waitKey(0)

    corner_binarized = ImageProcessing.automaticBinarization(corner_zoom)

    return corner_binarized


def getRankAndSuit(corner):
    # Get rank position on corner
    rank = corner[20:185, 0:128]
    # Get suit position on corner
    suit = corner[160:336, 0:128]

    # cv.imshow('Rank', rank)
    # cv.waitKey(0)
    # cv.imshow('Suit', suit)
    # cv.waitKey(0)

    # Resize them to standard dimensions
    rank_sized = ImageProcessing.findBiggestContourAndResizeByBoundingRectangle(rank, RANK_WIDTH, RANK_HEIGHT)
    suit_sized = ImageProcessing.findBiggestContourAndResizeByBoundingRectangle(suit, SUIT_WIDTH, SUIT_HEIGHT)

    cv.imshow('Rank', rank_sized)
    # cv.waitKey(0)
    cv.imshow('Suit', suit_sized)
    cv.waitKey(0)

    return rank_sized, suit_sized


def transformTemplateCards():
    for rank in ranks:
        for suit in suits:
            cardLabel = rank + "_of_" + suit
            template = cv.imread(
                "C:\\Users\\dariu\\PycharmProjects\\CardDetection\\images\\cards\\" + cardLabel + ".jpg", cv.IMREAD_GRAYSCALE)

            if template is None:
                print("No template found")
                return -1

            template_sized = cv.resize(template, (CARD_WIDTH, CARD_HEIGHT))
            corner = getCorner(template_sized)
            rank_sized, suit_sized = getRankAndSuit(corner)
            cv.imwrite("C:\\Users\\dariu\\PycharmProjects\\CardDetection\\images\\template\\rank\\" + rank + ".jpg", rank_sized)
            cv.imwrite("C:\\Users\\dariu\\PycharmProjects\\CardDetection\\images\\template\\suit\\" + suit + ".jpg", suit_sized)


def match(path, image, items):

    best_match = ""
    best_match_value = -1.0
    best_match_loc = []

    for item in items:
        template = cv.imread(path + item + ".jpg", cv.IMREAD_GRAYSCALE)

        if template is None:
            print("No template found")
            return -1

        res = cv.matchTemplate(image, template, cv.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

        match_loc = max_loc
        match_value = max_val

        if match_value > best_match_value:
            best_match_value = match_value
            best_match = item
            best_match_loc = match_loc

    return best_match


def matchRank(rank):
    res = match("C:\\Users\\dariu\\PycharmProjects\\CardDetection\\images\\template\\rank\\", rank, ranks)
    return res


def matchSuit(suit):
    res = match("C:\\Users\\dariu\\PycharmProjects\\CardDetection\\images\\template\\suit\\", suit, suits)
    return res


def getAllCardFromImage(image):
    preprocessed_image = ImageProcessing.preprocessImage(image)

    card_contours = getCardContours(preprocessed_image)

    cards = []

    for card_contour in card_contours:
        warp = extractCard(image, card_contour)
        corner = getCorner(warp)
        card_rank, card_suit = getRankAndSuit(corner)
        card = Card()
        card.rank = matchRank(card_rank)
        card.suit = matchSuit(card_suit)
        cards.append(card)

    return cards
