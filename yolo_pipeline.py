import cv2
import numpy as np
from ultralytics import YOLO

def order_corners(points): #takes (x, y) pairs

    points = np.array(points, dtype="float32")
    ordered = np.zeros((4, 2), dtype="float32")

    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).flatten()

    ordered[0] = points[np.argmin(sums)] # top left - smallest sum
    ordered[1] = points[np.argmin(diffs)] # top right - smallest diff
    ordered[2] = points[np.argmax(sums)] # bottom right - largest sum
    ordered[3] = points[np.argmax(diffs)]   # bottom left - largest diff

    return ordered

model_path="best.pt"
model = YOLO(model_path) # loads trained model into memory

def detect_doc(img_path, model, conf=0.25): # only trust detections if the model is at least 25% confident

    img = cv2.imread(img_path) # loads the actual photo as a np array (pixel data)

    results = model.predict(source=img_path, conf=conf, verbose=False) # verbose false turns off extra log messages to keep console clean

    if len(results[0].obb) == 0: # results[0].obb holds all the detected obb for the image
       # raise ValueError("No document detected")
        return None, img, None

    obb = results[0].obb #gets all detected boxes' conf scores
    best_idc = obb.conf.argmax() # finds the index of hte highest conf score

    corners = obb.xyxyxyxy[best_idc].cpu().numpy()
    # pulls out 4 corner coords for specific detection
    # .cpu() moves data from gpu memory to cpu memory
    # .numpy() converts corners from a pytorch tensor into plain np array
    confidence = float(obb.conf[best_idc].cpu().numpy())

    ordered = order_corners(corners)

    return ordered, img, confidence # we'll need raw image data for warp set

def perspective_warp(img, ordered_corners): # takes the original image and the ordered corners and returns a straightened top down view

    (tl, tr, br, bl) = ordered_corners # unpacks the 4 coords into individual variables

    # np.linalg.norm(point a - point b) calculates the straight line distance between two points
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = int(max(width_top, width_bottom)) # take the longer width so nothing gets cut off

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = int(max(height_left, height_right))

    outline = np.array([ # where we want each corner to end up in the final shape
        [0, 0], # top left
        [max_width - 1,0], # top right
        [max_width - 1, max_height - 1], # bottom right
        [0, max_height - 1] # bottom left
    ], dtype="float32")

    transformation_matrix = cv2.getPerspectiveTransform(ordered_corners, outline)
    # opencv looks at where the 4 point currently are - ordered_corners (tilted original image)
    # and where they need to end up - outline (flat rectangle)
    # calculates a transformation matrix

    warped = cv2.warpPerspective(img, transformation_matrix, (max_width, max_height))
    #applies the matrix to the whole image
    # final straightened output is cropped just to the document area (max width * max height)

    return warped


def run_image(img_path, model, output_path=None):
    if output_path is None:
        output_path = f"images/{img_path}_warped.png"

    corners, img, confidence = detect_doc(img_path, model)

    if confidence is not None:
        print(f"Detection confidence: {confidence:.3f}")
        if confidence < 0.5: print("Needs review")


    if corners is not None:
        print("Corners successfully detected")
        processed_img = perspective_warp(img, corners)
    else:
        print("No document detected - using raw image")
        processed_img = img


    # warped = perspective_warp(img, corners)
    # cv2.imwrite(output_path, warped)

    return processed_img, confidence