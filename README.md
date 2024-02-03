# ID Card Detection

Egyptian National ID Card Detection, Alignment and Text Region Extractions. 

## Assumptions: 

### Assumption 1: 

This solution assumes the ID is scanned. 

### Assumption 2: 

This solution assumes the detection of the front side only of the ID.


## Demo: 

Input image as follows: 

![test](https://github.com/KarimIbrahim11/ID-Card-Detection/assets/47744559/e5847776-90f1-4e19-aa57-85e70be373c7)

The script will first detect the SIFT features and match them between the following template and the test image:

![sift_template](https://github.com/KarimIbrahim11/ID-Card-Detection/assets/47744559/6958d77d-ae5f-4297-86c4-3ac737f507f0)

If Matches Found: 

![Figure_1](https://github.com/KarimIbrahim11/ID-Card-Detection/assets/47744559/8e41f3d6-d7bd-477d-84b8-5c4b592c855d)

A homography matrix between the image and the template is found. If the image moves out of frame, the matrix is compensated
by another translation matrix to accomodate the translation back in to frame. The result of the translation is as follows: 

![Warped](https://github.com/KarimIbrahim11/ID-Card-Detection/assets/47744559/c05626f1-ebbe-4d0b-ae67-40480946c42d)

After warping, Canny Edges are applied for edge detection and finding the largest contour to only select the card from the scene.
The largest contour is then warped to a fixed size of (672, 448). This is a static size I configured best for the application.

![cropped_card](https://github.com/KarimIbrahim11/ID-Card-Detection/assets/47744559/710b0ab8-3211-4e8b-8375-6662870911f3)

The card is then converted to binary, dilated and eroded for blob detection by finding contours around blobs. The blobs will resemble
the text regions and the face. We need to disregard the face Bounding Box. I used dlib's face detector to find the region of the face
and delete any of the boxes that overlap with the region of the face. The final result is as follows: 

![test](https://github.com/KarimIbrahim11/ID-Card-Detection/assets/47744559/d503cff5-986e-4dc6-b46d-2c3d38f6abb5)

The bounding boxes of the text are saved in ```output/``` directory


## To Duplicate Results: 

Add yout images to the test-dataset directory and write the following commands
```
git clone https://github.com/KarimIbrahim11/ID-Card-Detection.git
cd ID-Card-Detection
conda env create -f env.yaml
conda activate env
python id_detector.py
```







