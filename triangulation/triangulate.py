import cv2
import numpy as np
import json
import math

from triangulation.detect import detect

CALIBRATION_FILE = "./extrinsic_calibration/extrinsic_calibration.json"


class DartScorer:
    def __init__(self):
        self._load_calibration(CALIBRATION_FILE)
        self._setup_dartboard_geometry()
        self.camera_name_to_id = {"left": 1, "mid": 0, "right": 2}

    def _load_calibration(self, calibration_file: str):
        with open(calibration_file, "r") as f:
            calib_data = json.load(f)

        self.cameras = {}

        # load intrinsic calibrations, used for triangulation later
        for cam_data in calib_data["cameras"]:
            cam_id = cam_data["camera_id"]
            K = np.array(cam_data["intrinsic_matrix"])
            R = np.array(cam_data["rotation_matrix"])
            t = np.array(cam_data["translation_vector"]).reshape(3, 1)
            P = K @ np.hstack([R, t])

            self.cameras[cam_id] = {
                "K": K,
                "dist": np.array(cam_data["distortion_coef"]),
                "P": P,
            }

    def _setup_dartboard_geometry(self):
        # normed dartboard meassurements
        self.geom = {
            "outer_double": 170.0,
            "inner_double": 162.0,
            "triple_outer": 107.0,
            "triple_inner": 99.0,
            "outer_bull": 16.0,
            "inner_bull": 6.35,
        }
        self.segments = [
            20,
            1,
            18,
            4,
            13,
            6,
            10,
            15,
            2,
            17,
            3,
            19,
            7,
            16,
            8,
            11,
            14,
            9,
            12,
            5,
        ]

    def _point_to_score(self, point_3d: np.ndarray) -> dict:
        x, y, _ = point_3d
        radius = math.sqrt(x**2 + y**2)
        angle_rad = math.atan2(x, y)
        angle_deg = math.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360

        # calculate +9 because the segment 20 is at the top and we want a wire to be the 0 value
        segment_index = int((angle_deg + 9) / 18) % 20
        base_score = self.segments[segment_index]

        if radius <= self.geom["inner_bull"]:
            return {"base_score": 25, "multiplier": "double bull"}
        if radius <= self.geom["outer_bull"]:
            return {"base_score": 25, "multiplier": "single bull"}
        if self.geom["triple_inner"] < radius <= self.geom["triple_outer"]:
            return {"base_score": base_score, "multiplier": "triple"}
        if self.geom["inner_double"] < radius <= self.geom["outer_double"]:
            return {"base_score": base_score, "multiplier": "double"}
        if radius > self.geom["outer_double"]:
            return {"base_score": 0, "multiplier": "outside"}

        return {"base_score": base_score, "multiplier": "single"}

    def calculate_throw(self, detection_dict: dict) -> dict:
        valid_points = {}
        for cam_name, points_list in detection_dict.items():
            cam_id = self.camera_name_to_id[cam_name]

            if points_list and len(points_list) > 0:
                point = points_list[0]
                if point is not None and len(point) == 2:
                    valid_points[cam_id] = np.array(point, dtype=np.float32)

        if len(valid_points) < 2:
            return {"success": False}

        cam_ids = list(valid_points.keys())
        undistorted_points = {}

        for cam_id in cam_ids:
            pt = valid_points[cam_id].reshape(1, 1, 2)
            cam = self.cameras[cam_id]
            undistorted_points[cam_id] = cv2.undistortPoints(
                pt, cam["K"], cam["dist"], P=cam["K"]
            ).reshape(2)

        all_3d_points = []
        for i in range(len(cam_ids)):
            for j in range(i + 1, len(cam_ids)):
                cam1_id, cam2_id = cam_ids[i], cam_ids[j]

                P1 = self.cameras[cam1_id]["P"]
                P2 = self.cameras[cam2_id]["P"]

                pt1 = undistorted_points[cam1_id]
                pt2 = undistorted_points[cam2_id]

                points_4d = cv2.triangulatePoints(P1, P2, pt1, pt2)
                point_3d = (points_4d[:3] / points_4d[3]).flatten()
                all_3d_points.append(point_3d)

        if not all_3d_points:
            return {"success": False}

        # mean of all triangulations
        final_point_3d = np.mean(all_3d_points, axis=0)
        score_info = self._point_to_score(final_point_3d)

        return {
            "success": True,
            "position": tuple(final_point_3d[:2]),
            "base_score": score_info["base_score"],
            "multiplier": score_info["multiplier"],
        }


if __name__ == "__main__":
    points = detect()
    scorer = DartScorer()
    result = scorer.calculate_throw(points)
    multiplier = result["multiplier"]
    base_score = result["base_score"]

    score = 0
    if multiplier == "single":
        score = base_score
    elif multiplier == "double":
        score = 2 * base_score
    elif multiplier == "triple":
        score = 3 * base_score
    elif multiplier == "single bull":
        score = base_score
    elif multiplier == "double bull":
        score = 2 * base_score

    print(f"Multiplier: {multiplier}, Base Score: {base_score}")
