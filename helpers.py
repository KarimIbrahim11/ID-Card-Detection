import numpy as np
import cv2
import dlib
import matplotlib.pyplot as plt


def applyOrientation(gray, img, M):
    # # Apply the homography to warp the image
    # temp_gray = gray.copy()
    # temp_gray = cv2.warpPerspective(temp_gray, M, (temp_gray.shape[1], temp_gray.shape[0]), flags=cv2.INTER_AREA,
    #                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    # cv2.imshow("warped_1", temp_gray)
    # cv2.waitKey(0)
    #
    # # Find the corners after the transform has been applied
    # height, width = temp_gray.shape[:2]
    height, width = gray.shape[:2]
    corners = np.array([
        [0, 0],
        [0, height - 1],
        [width - 1, height - 1],
        [width - 1, 0]
    ])
    corners = cv2.perspectiveTransform(np.float32([corners]), M)[0]
    # Find the bounding rectangle
    bx, by, bwidth, bheight = cv2.boundingRect(corners)

    if bx < 0 and by < 0:
        translation = np.float32([[1, 0, -bx], [0, 1, -by], [0, 0, 1]])
    elif bx < 0 <= by:
        translation = np.float32([[1, 0, -bx], [0, 1, 0], [0, 0, 1]])
    elif by < 0 <= bx:
        translation = np.float32([[1, 0, 0], [0, 1, -by], [0, 0, 1]])
    else:
        translation = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    M = translation.dot(M)

    img = cv2.warpPerspective(img, M, (bwidth, bheight), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(255, 255, 255))
    gray = cv2.warpPerspective(gray, M, (bwidth, bheight), flags=cv2.INTER_AREA,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(255, 255, 255))
    # cv2.imshow("Warped: ", img)
    # cv2.waitKey(0)
    return gray, img, M


def findOrientation(gray):
    MIN_MATCH_COUNT = 8
    img1 = cv2.imread('sift_template.jpg', cv2.IMREAD_GRAYSCALE)  # queryImage
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(gray, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        gray = cv2.polylines(gray, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None
        return None, None
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    # img3 = cv2.drawMatches(img1, kp1, gray, kp2, good, None, **draw_params)
    # plt.imshow(img3, 'gray')
    # plt.title('SIFT Features Matches')
    # plt.show()
    return M, [np.int32(dst)]


def findLargestContour(edges):
    # Taking a matrix of size 5 as the kernel
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.erode(edges, kernel, iterations=2)
    # Hough for line detection
    _, thresh = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)
    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # max contour is probably the ID
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the coordinates of the top-leftmost point
        top_leftmost_point = largest_contour[np.argmin(largest_contour[:, 0, 0] + largest_contour[:, 0, 1])][0]
        top_rightmost_point = largest_contour[np.argmax(largest_contour[:, 0, 0] - largest_contour[:, 0, 1])][0]
        bot_leftmost_point = largest_contour[np.argmax(largest_contour[:, 0, 0] + largest_contour[:, 0, 1])][0]
        bot_rightmost_point = largest_contour[np.argmin(largest_contour[:, 0, 0] - largest_contour[:, 0, 1])][0]

        return [top_leftmost_point, top_rightmost_point, bot_leftmost_point, bot_rightmost_point], largest_contour
    else:
        return None, _


def blur_img(gray):
    sigma = 1.5
    kernel_size = (5, 5)
    # Apply Gaussian blur to the gray image
    gray = cv2.GaussianBlur(gray, kernel_size, sigma)
    return gray


def output_img(img, largest_contour):
    # Get Bounding Points the old way
    top_leftmost_point = largest_contour[np.argmin(largest_contour[:, 0, 0] + largest_contour[:, 0, 1])][0]
    top_rightmost_point = largest_contour[np.argmax(largest_contour[:, 0, 0] - largest_contour[:, 0, 1])][0]
    bot_leftmost_point = largest_contour[np.argmax(largest_contour[:, 0, 0] + largest_contour[:, 0, 1])][0]
    bot_rightmost_point = largest_contour[np.argmin(largest_contour[:, 0, 0] - largest_contour[:, 0, 1])][0]
    # Define reference points for the desired orientation (e.g., corners of a rectangle)
    reference_points = np.array([[0, 0], [672, 0], [672, 448], [0, 448]])
    source_points = np.array([top_leftmost_point, top_rightmost_point, bot_leftmost_point, bot_rightmost_point])
    # Find the perspective transformation matrix using cv2.findHomography
    transform_matrix, _ = cv2.findHomography(source_points, reference_points)
    # Apply the perspective transformation to the original image and return
    return cv2.warpPerspective(img, transform_matrix, (672, 448))


def detect_text_regions(im_gray, im):
    ret, thresh = cv2.threshold(im_gray, 127, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    kernel3 = np.array([[0, 0, 0],
                        [1, 1, 1],
                        [0, 0, 0]], dtype=np.uint8)

    thresh = cv2.dilate(thresh, kernel3, iterations=20)
    # cv2.imshow("bin", thresh)
    # cv2.waitKey()
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    minArea = 1500  # nothing
    bounding_boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if (area > minArea):
            x, y, w, h = cv2.boundingRect(cnt)
            bounding_boxes.append([x, y, x + w, y + h])
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

    bounding_boxes = np.array(bounding_boxes)
    x, y, w, h = face_bbox(im)
    reference_box = x, y, x + w, y + h
    bounding_boxes = remove_overlapping_boxes(bounding_boxes, reference_box, 0.01)

    # Apply non-maximum suppression
    selected_boxes = non_max_suppression(bounding_boxes)

    return selected_boxes, bounding_boxes


# Function for non-maximum suppression
def non_max_suppression(boxes, overlap_threshold=0.3):
    if len(boxes) == 0:
        return []

    # Extract coordinates of bounding boxes
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    # Compute the area of each bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Sort the bounding boxes by their scores (you may want to sort by area, confidence, etc.)
    idxs = np.argsort(y2)

    # Initialize the list to store the final, non-overlapping bounding boxes
    pick = []

    while len(idxs) > 0:
        # Select the last bounding box in the sorted list
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Find the overlap with other bounding boxes
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Compute the width and height of the overlap region
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute the overlap ratio
        overlap = (w * h) / area[idxs[:last]]

        # Delete the indices of bounding boxes with overlap greater than the threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))

    # Return the indices of the selected bounding boxes
    return pick


def remove_overlapping_boxes(bounding_boxes, reference_box, overlap_threshold=0.3):
    x1, y1, x2, y2 = reference_box

    # Compute the width and height of the reference bounding box
    w_ref = x2 - x1 + 1
    h_ref = y2 - y1 + 1

    # Calculate overlap with other bounding boxes
    overlap = (np.maximum(0, np.minimum(x2, bounding_boxes[:, 2]) - np.maximum(x1, bounding_boxes[:, 0]) + 1) *
               np.maximum(0, np.minimum(y2, bounding_boxes[:, 3]) - np.maximum(y1, bounding_boxes[:, 1]) + 1)) / (
                      w_ref * h_ref)

    # Find indices of bounding boxes that don't overlap with the reference box
    indices_to_keep = np.where(overlap <= overlap_threshold)[0]

    # Return the remaining bounding boxes
    return bounding_boxes[indices_to_keep]


def face_bbox(gray):
    face_detector = dlib.get_frontal_face_detector()
    # face detecting
    faces = face_detector(gray)
    if not faces:
        print("No faces found :( ")
    else:
        return faces[0].left(), faces[0].top(), faces[0].width(), faces[0].height()
