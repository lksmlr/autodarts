import cv2
import numpy as np
import glob
import os
import json


# count checkerboard lines
checkerboard = (9, 6)

objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : checkerboard[0], 0 : checkerboard[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []

image_dir = os.path.join(os.path.dirname(__file__), "./calibration_images")
image_paths = glob.glob(os.path.abspath(os.path.join(image_dir, "*.jpg")))
gray_shape = None


for fname in image_paths:
    img = cv2.imread(fname)
    if img is None:
        print(f"Fehler beim Laden des Bildes: {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, checkerboard, None)

    objpoints.append(objp)
    imgpoints.append(corners)
    gray_shape = gray.shape[::-1]


if not objpoints:
    raise ValueError("Keine gültigen Schachbretter gefunden.")


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray_shape, None, None
)


intrinsic_matrix = mtx.tolist()
distortion_coef = dist.ravel().tolist()

formatted = {
    "intrinsic_matrix": intrinsic_matrix,
    "distortion_coef": distortion_coef,
    "rotation": 0,
}

print("\nKameraparameter:")
print(json.dumps(formatted, indent=4))
