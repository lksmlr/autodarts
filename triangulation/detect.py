import cv2
from ultralytics import YOLO
import math


model = YOLO("./models/model_v1.pt")

# cam 1: mid, cam 0: left, cam 2: right
cam_ids = [1, 0, 2]
cam_map = ["mid", "left", "right"]


def filter(points: list[tuple[int, int]], threshold: int = 5) -> list[tuple[int, int]]:
    result = []
    for p in points:
        too_close = False
        for r in result:
            distance = math.dist(p, r)  # euclidian distance
            if distance < threshold:
                too_close = True
                break
        if not too_close:
            result.append(p)
    return result


def detect():
    detections = {}
    for cam_id, mapping in zip(cam_ids, cam_map):
        capture = cv2.VideoCapture(cam_id)
        _, frame = capture.read()

        result = model(frame, conf=0.04)

        if result[0].keypoints.xy.numel() > 0:
            points = [
                (int(pt[0][0]), int(pt[0][1])) for pt in result[0].keypoints.xy.tolist()
            ]
            detections[mapping] = filter(points)

        else:
            detections[mapping] = []

        capture.release()

    print(detections)

    return detections


if __name__ == "__main__":
    detect()
