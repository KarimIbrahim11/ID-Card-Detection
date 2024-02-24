import os
from helpers import *

if __name__ == '__main__':
    directory_path = "test-dataset"

    # Traverse through the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            print("filename:", filename)
            # Read the image using OpenCV
            image_path = os.path.join(directory_path, filename)
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Get homography between Gomhoreyet Masr template and ID
            M, match_boundingbox = findOrientation(gray)

            # If matches found Apply Orientation
            if M is not None:
                gray, img, M = applyOrientation(gray, img, M)

                # blur gray for smoothing the image for the edge detection
                gray = blur_img(gray)
                # Adaptive threshold
                bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                # Edge detection
                edges = cv2.Canny(bin, 200, 600, apertureSize=3)

                # cv2.imshow('edges through Canny', edges)
                # cv2.waitKey(0)

                # Getting largest Contour
                contour_bbox, largest_contour = findLargestContour(edges)
                if contour_bbox is not None:
                    for point in contour_bbox:
                        cv2.circle(img, point, thickness=5, color=(255, 0, 0), radius=3)

                    # cv2.imshow("Largest Contour:", img)
                    # cv2.waitKey(0)

                    cropped_img = output_img(img, largest_contour)
                    cropped_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

                    output_pth = 'output/' + filename
                    print(output_pth)

                    selections, bboxes = detect_text_regions(cropped_gray, cropped_img)

                    for i in selections:
                        x, y, x2, y2 = bboxes[i]
                        cv2.imwrite(f'output/{i}_{filename}', cropped_img[y:y2, x:x2])
                        cv2.rectangle(cropped_img, (x, y), (x2, y2), (0, 255, 0), 2)

                    cv2.imshow("Final Result:", cropped_img)
                    cv2.imwrite(output_pth, cropped_img)
                    cv2.waitKey(0)
                else:
                    print("No contours")
            else:
                print("No ID Card Found / No contours")
