import cv2
import numpy as np
import json
from pathlib import Path


class DartboardCalibrator:
    def __init__(self):
        # Standard Dartscheiben Koordinaten in mm
        # Double-Ring äußerer Radius: 170mm, Bullseye: 6.35mm
        self.world_points = np.array(
            [
                [0.0, 170.0, 0.0],  # 12 Uhr (oben)
                [170.0, 0.0, 0.0],  # 3 Uhr (rechts)
                [0.0, -170.0, 0.0],  # 6 Uhr (unten)
                [-170.0, 0.0, 0.0],  # 9 Uhr (links)
                [0.0, 0.0, 0.0],  # Bullseye (Zentrum)
            ],
            dtype=np.float32,
        )

        self.point_names = [
            "12 Uhr (oben)",
            "3 Uhr (rechts)",
            "6 Uhr (unten)",
            "9 Uhr (links)",
            "Bullseye",
        ]
        self.colors = [
            (0, 255, 0),
            (255, 0, 0),
            (0, 255, 255),
            (255, 255, 0),
            (255, 0, 255),
        ]

        self.image_points = []
        self.images = []
        self.original_images = []  # Zum Speichern der Originalaufnahmen für den Reset
        self.current_camera = 0
        self.current_point = 0
        self.camera_matrices = []
        self.dist_coeffs = []

    def load_intrinsic_parameters(self, camera_params_files):
        """Lädt die intrinsischen Parameter für alle Kameras"""
        for param_file in camera_params_files:
            with open(param_file, "r") as f:
                params = json.load(f)
                self.camera_matrices.append(np.array(params["intrinsic_matrix"]))
                self.dist_coeffs.append(np.array(params["distortion_coef"]))
                print(f"Geladen: {param_file}")
                print(
                    f"  - Brennweite: fx={params['intrinsic_matrix'][0][0]:.1f}, fy={params['intrinsic_matrix'][1][1]:.1f}"
                )
                print(
                    f"  - Hauptpunkt: cx={params['intrinsic_matrix'][0][2]:.1f}, cy={params['intrinsic_matrix'][1][2]:.1f}"
                )
                print(f"  - Distortion: k1={params['distortion_coef'][0]:.3f}")

    def capture_images_from_cameras(self, cam_ids):
        """
        NEU: Erfasst Bilder von den angegebenen Kameras.
        Zeigt einen Live-Feed an und wartet auf Tastendruck zur Aufnahme.
        """
        captured_images = []
        print("Starte Bilderfassung...")

        for cam_id in cam_ids:
            cap = cv2.VideoCapture(
                cam_id, cv2.CAP_DSHOW
            )  # CAP_DSHOW kann unter Windows die Initialisierung beschleunigen
            if not cap.isOpened():
                print(f"FEHLER: Kamera mit ID {cam_id} konnte nicht geöffnet werden.")
                continue

            window_name = f"Live-Vorschau Kamera {cam_id}"
            print(f"\nKamera {cam_id} geöffnet. Live-Vorschau wird angezeigt.")
            print(
                ">>> Drücken Sie 's', um ein Bild aufzunehmen, oder 'q' zum Abbrechen."
            )

            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"FEHLER: Konnte keinen Frame von Kamera {cam_id} lesen.")
                    break

                # Anweisungen auf dem Bild anzeigen
                cv2.putText(
                    frame,
                    "Druecke 's' zum Aufnehmen | 'q' zum Abbrechen",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    ret, image_to_save = cap.read()  # Finalen, sauberen Frame lesen
                    if ret:
                        captured_images.append(image_to_save)
                        print(f"Bild von Kamera {cam_id} erfolgreich aufgenommen.")
                    else:
                        print(f"FEHLER bei der finalen Aufnahme von Kamera {cam_id}.")
                    break
                elif key == ord("q"):
                    print("Bilderfassung durch Benutzer abgebrochen.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return []  # Leere Liste zurückgeben, um Abbruch zu signalisieren

            cap.release()
            cv2.destroyWindow(window_name)

        return captured_images

    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback für Punktmarkierung"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_point < len(self.point_names):
                self.image_points[self.current_camera].append([x, y])

                color = self.colors[self.current_point]
                cv2.circle(self.images[self.current_camera], (x, y), 5, color, -1)
                cv2.putText(
                    self.images[self.current_camera],
                    f"P{self.current_point + 1}",
                    (x + 10, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

                print(
                    f"Kamera {self.current_camera + 1}: {self.point_names[self.current_point]} markiert bei ({x}, {y})"
                )
                self.current_point += 1

                if self.current_point >= len(self.point_names):
                    print(f"Alle Punkte für Kamera {self.current_camera + 1} markiert!")
                    print("Drücken Sie 'n' für nächste Kamera oder 'c' zum Kalibrieren")
                else:
                    print(f"Nächster Punkt: {self.point_names[self.current_point]}")

    def calibrate_camera(self, camera_idx):
        """Kalibriert eine einzelne Kamera (extrinsische Parameter)"""
        if len(self.camera_matrices) <= camera_idx:
            print(f"Keine intrinsischen Parameter für Kamera {camera_idx + 1} geladen!")
            return None, None, None

        image_points = np.array(self.image_points[camera_idx], dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(
            self.world_points,
            image_points,
            self.camera_matrices[camera_idx],
            self.dist_coeffs[camera_idx],
        )

        if success:
            R, _ = cv2.Rodrigues(rvec)

            projected_points, _ = cv2.projectPoints(
                self.world_points,
                rvec,
                tvec,
                self.camera_matrices[camera_idx],
                self.dist_coeffs[camera_idx],
            )

            error = cv2.norm(
                image_points, projected_points.reshape(-1, 2), cv2.NORM_L2
            ) / len(image_points)

            # Diese Ausgaben sind jetzt im save_calibration Teil, um Doppelung zu vermeiden
            return R, tvec, error
        else:
            print(f"Kalibrierung für Kamera {camera_idx + 1} fehlgeschlagen!")
            return None, None, None

    def save_calibration(self, output_file):
        """Speichert alle Kalibrierungsdaten"""
        calibration_data = {"world_points": self.world_points.tolist(), "cameras": []}

        print("\n=== Starte finale Kalibrierung und Speicherung ===")
        for i in range(len(self.images)):
            if len(self.image_points[i]) == len(self.point_names):
                R, tvec, error = self.calibrate_camera(i)
                if R is not None:
                    rvec, _ = cv2.Rodrigues(R)

                    print(f"\n--- Ergebnisse für Kamera {i + 1} ---")
                    print(f"Reprojection Error: {error:.2f} pixels")
                    print(f"Translation (tvec) [mm]: {tvec.flatten()}")
                    print(f"Rotation (rvec) [rad]: {rvec.flatten()}")

                    camera_data = {
                        "camera_id": i,
                        "image_points": self.image_points[i],
                        "rotation_matrix": R.tolist(),
                        "rotation_vector": rvec.flatten().tolist(),
                        "translation_vector": tvec.flatten().tolist(),
                        "reprojection_error": error,
                        "intrinsic_matrix": self.camera_matrices[i].tolist()
                        if i < len(self.camera_matrices)
                        else None,
                        "distortion_coef": self.dist_coeffs[i].tolist()
                        if i < len(self.dist_coeffs)
                        else None,
                    }
                    calibration_data["cameras"].append(camera_data)

        # Sicherstellen, dass der Ordner existiert
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(calibration_data, f, indent=4)

        print(f"\nKalibrierung erfolgreich gespeichert in: {output_file}")

    def run_calibration(
        self,
        images_to_process,
        intrinsic_params_files=None,
        output_file="dartboard_calibration.json",
    ):
        """
        Hauptfunktion für die Kalibrierung.
        MODIFIZIERT: Akzeptiert Bild-Arrays statt Dateipfaden.
        """
        if intrinsic_params_files:
            self.load_intrinsic_parameters(intrinsic_params_files)

        # Bilder und Punktelisten initialisieren
        for img in images_to_process:
            self.images.append(img.copy())
            self.original_images.append(img.copy())  # Kopie für Reset speichern
            self.image_points.append([])

        if not self.images:
            print("Keine Bilder zum Kalibrieren vorhanden!")
            return

        print(f"\n{len(self.images)} Bilder sind bereit für die Kalibrierung.")
        print("Anleitung:")
        print("- Klicken Sie die Punkte in folgender Reihenfolge:")
        for i, name in enumerate(self.point_names):
            print(f"  {i + 1}. {name}")
        print("- 'n': Nächste Kamera (wenn alle Punkte gesetzt sind)")
        print("- 'r': Markierungen für die aktuelle Kamera zurücksetzen")
        print(
            "- 'c': Kalibrierung starten (wenn alle Punkte für alle Kameras gesetzt sind)"
        )
        print("- 'q': Beenden")

        # Kalibrierungsprozess starten
        all_points_marked = False
        while self.current_camera < len(self.images):
            camera_idx = self.current_camera
            window_name = f"Kamera {camera_idx + 1} - Kalibrierung"

            # Nur beim ersten Mal für die Kamera anzeigen
            if not self.image_points[camera_idx]:
                print(f"\n=== Kamera {camera_idx + 1}/{len(self.images)} ===")
                print(f"Bitte markieren Sie: {self.point_names[self.current_point]}")

            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(window_name, self.mouse_callback)

            while True:
                cv2.imshow(window_name, self.images[camera_idx])
                key = cv2.waitKey(1) & 0xFF

                # Nächste Kamera
                if key == ord("n") and self.current_point >= len(self.point_names):
                    cv2.destroyWindow(window_name)
                    self.current_camera += 1
                    self.current_point = 0
                    break

                # Reset
                elif key == ord("r"):
                    self.image_points[camera_idx] = []
                    self.current_point = 0
                    # MODIFIZIERT: Bild aus der gespeicherten Kopie wiederherstellen
                    self.images[camera_idx] = self.original_images[camera_idx].copy()
                    print(f"Markierungen für Kamera {camera_idx + 1} zurückgesetzt.")
                    print(f"Bitte markieren Sie: {self.point_names[0]}")

                # Kalibrierung starten und Schleife beenden
                elif key == ord("c"):
                    if all(
                        len(pts) == len(self.point_names) for pts in self.image_points
                    ):
                        all_points_marked = True
                        break
                    else:
                        print(
                            "Fehler: Bitte markieren Sie zuerst alle 5 Punkte auf ALLEN Kamerabildern."
                        )

                # Beenden
                elif key == ord("q"):
                    cv2.destroyAllWindows()
                    print("Kalibrierung abgebrochen.")
                    return

            if all_points_marked:
                cv2.destroyAllWindows()
                break  # Äußere Schleife verlassen, um zur Speicherung überzugehen

        # Kalibrierung durchführen und speichern
        if all_points_marked:
            self.save_calibration(output_file)
        else:
            print(
                "Nicht alle Punkte wurden markiert. Kalibrierung wird nicht gespeichert."
            )


def main():
    cam_ids = [1, 0, 2]

    intrinsic_files = [
        "../intrinsic_calibration/mid_intrinsic_calibration.json",
        "../intrinsic_calibration/left_intrinsic_calibration.json",
        "../intrinsic_calibration/right_intrinsic_calibration.json",
    ]

    calibrator = DartboardCalibrator()

    # 1. Bilder von den Kameras aufnehmen
    captured_images = calibrator.capture_images_from_cameras(cam_ids)

    # 2. Prüfen, ob die Aufnahme erfolgreich war (d.h. nicht abgebrochen wurde)
    if not captured_images or len(captured_images) != len(cam_ids):
        print("\nKalibrierung abgebrochen, da nicht alle Bilder erfasst wurden.")
        return

    # 3. Kalibrierung mit den aufgenommenen Bildern starten
    calibrator.run_calibration(
        captured_images,
        intrinsic_files,
        "extrinsic_calibration.json",
    )


if __name__ == "__main__":
    main()
