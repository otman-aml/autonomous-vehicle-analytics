"""
Vehicle Detection, Tracking and Analytics — Autonomous Vehicles Context
=======================================================================
Pipeline : YOLOv8 (detection) → ByteTrack (tracking) → ROI → Speed (px/s) → Dashboard

Usage:
    python vehicle_analytics.py --video data/demo_short.mp4 --model yolov8s.pt
    python vehicle_analytics.py --video data/demo_short.mp4 --no-display

Author: OTMAN AMAL
Faculté Polydisciplinaire de Khouribga (FPK) — M2 SIIA
"""

import argparse
import json
import math
import os
from collections import defaultdict

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


# ============================================================
# 1) CONFIGURATION PAR DÉFAUT
# ============================================================
DEFAULT_MODEL = "yolov8s.pt"
DEFAULT_POLYGONS = "data/polygons.json"
DEFAULT_OUTPUT = "outputs/output_lane_detection.mp4"

DETECTION_REGION_FALLBACK = {"x": 340, "y": 220, "width": 340, "height": 180}


# ============================================================
# 2) LANE DETECTOR
# ============================================================
class LaneDetector:
    def __init__(self, json_path: str, video_path: str) -> None:
        self.lane_polygons = {}
        self.lane_names = []
        self.detection_zone = None
        self.vehicle_counts = defaultdict(int)
        self.lane_speeds = defaultdict(list)
        self.lane_average_speeds = defaultdict(float)
        self.video_path = video_path
        self.json_path = json_path

        try:
            self.load_polygons()
        except FileNotFoundError:
            print(f"⚠️  {json_path} introuvable → ROI par défaut utilisé")
            self._build_fallback_zone()
        except (ValueError, KeyError) as e:
            print(f"⚠️  Erreur parsing JSON ({e}) → ROI par défaut utilisé")
            self._build_fallback_zone()

    def _build_fallback_zone(self) -> None:
        cap = cv2.VideoCapture(self.video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        r = DETECTION_REGION_FALLBACK
        self.detection_zone = np.array([
            [r["x"], r["y"]],
            [r["x"] + r["width"], r["y"]],
            [r["x"] + r["width"], r["y"] + r["height"]],
            [r["x"], r["y"] + r["height"]],
        ], dtype=np.int32)
        self.lane_polygons["road"] = self.detection_zone
        self.lane_names = ["road"]

    def load_polygons(self) -> None:
        with open(self.json_path, "r") as f:
            data = json.load(f)

        boxes = data.get("boxes", [])
        if not boxes:
            raise ValueError("No 'boxes' found in JSON")

        original_width = data.get("width", 504)
        original_height = data.get("height", 354)

        cap = cv2.VideoCapture(self.video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        scale_x = width / original_width
        scale_y = height / original_height

        translation = {
            "emniyet_seridi": "emergency_lane",
            "sag_serit": "right_lane",
            "orta_serit1": "middle_lane1",
            "orta_serit2": "middle_lane2",
            "orta_seri2": "middle_lane2",
            "sol_serit": "left_lane",
            "yol": "road",
        }

        all_points = []

        for box in boxes:
            label = box.get("label", "unknown")
            box_id = str(box.get("id", ""))
            label = translation.get(label, label)

            if "points" in box:
                scaled_pts = [
                    [p[0] * scale_x, p[1] * scale_y]
                    for p in box["points"] if isinstance(p, list) and len(p) == 2
                ]
                all_points.extend(scaled_pts)

                if box_id == "7":  # road names box → ignoré
                    continue

                if not scaled_pts:
                    continue

                if label in self.lane_polygons:
                    counter = 1
                    new_label = f"{label}_{counter}"
                    while new_label in self.lane_polygons:
                        counter += 1
                        new_label = f"{label}_{counter}"
                    label = new_label

                self.lane_polygons[label] = np.array(scaled_pts, dtype=np.int32)
                if label not in self.lane_names:
                    self.lane_names.append(label)

            elif all(k in box for k in ["x", "y", "width", "height"]):
                x = float(box["x"]) * scale_x
                y = float(box["y"]) * scale_y
                w = float(box["width"]) * scale_x
                h = float(box["height"]) * scale_y
                rect = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                all_points.extend(rect)

                if box_id == "7":
                    continue

                if label in self.lane_polygons:
                    counter = 1
                    new_label = f"{label}_{counter}"
                    while new_label in self.lane_polygons:
                        counter += 1
                        new_label = f"{label}_{counter}"
                    label = new_label

                self.lane_polygons[label] = np.array(rect, dtype=np.int32)
                if label not in self.lane_names:
                    self.lane_names.append(label)

        # Zone de détection : id=6 si présent, sinon enveloppe convexe
        for box in boxes:
            if str(box.get("id")) == "6" and "points" in box:
                scaled = [
                    [p[0] * scale_x, p[1] * scale_y]
                    for p in box["points"] if isinstance(p, list) and len(p) == 2
                ]
                if scaled:
                    self.detection_zone = np.array(scaled, dtype=np.int32)
                    break

        if self.detection_zone is None:
            if all_points:
                self.detection_zone = cv2.convexHull(np.array(all_points, dtype=np.int32))
            else:
                r = DETECTION_REGION_FALLBACK
                self.detection_zone = np.array([
                    [r["x"] * scale_x, r["y"] * scale_y],
                    [(r["x"] + r["width"]) * scale_x, r["y"] * scale_y],
                    [(r["x"] + r["width"]) * scale_x, (r["y"] + r["height"]) * scale_y],
                    [r["x"] * scale_x, (r["y"] + r["height"]) * scale_y],
                ], dtype=np.int32)

        print(f"✅ {len(self.lane_polygons)} polygones chargés : {', '.join(self.lane_names)}")

    def determine_lane(self, point):
        for lane_name, polygon in self.lane_polygons.items():
            if cv2.pointPolygonTest(polygon, point, False) >= 0:
                return lane_name
        return "unknown"

    def is_in_detection_zone(self, bbox):
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        return cv2.pointPolygonTest(self.detection_zone, (cx, cy), False) >= 0

    def count_vehicle(self, lane_name):
        self.vehicle_counts[lane_name] += 1

    def reset_counts(self):
        self.vehicle_counts = defaultdict(int)

    def add_vehicle_speed(self, lane_name, speed):
        if speed > 0:
            self.lane_speeds[lane_name].append(speed)
            avg = sum(self.lane_speeds[lane_name]) / len(self.lane_speeds[lane_name])
            self.lane_average_speeds[lane_name] = round(avg, 2)


# ============================================================
# 3) TRAFFIC FLOW MANAGER
# ============================================================
class DetectionsManager:
    def __init__(self) -> None:
        self.tracker_id_to_zone_id = {}
        self.counts = {}

    def update(self, detections_all, detections_in_zones, detections_out_zones):
        for zone_in_id, detections_in_zone in enumerate(detections_in_zones):
            for tracker_id in detections_in_zone.tracker_id:
                self.tracker_id_to_zone_id.setdefault(tracker_id, zone_in_id)

        for zone_out_id, detections_out_zone in enumerate(detections_out_zones):
            for tracker_id in detections_out_zone.tracker_id:
                zone_in_id = self.tracker_id_to_zone_id.get(tracker_id, 0)
                self.counts.setdefault(zone_out_id, {})
                self.counts[zone_out_id].setdefault(zone_in_id, set())
                self.counts[zone_out_id][zone_in_id].add(tracker_id)

        if len(detections_all) > 0:
            detections_all.class_id = np.vectorize(
                lambda x: self.tracker_id_to_zone_id.get(x, 0)
            )(detections_all.tracker_id)
        else:
            detections_all.class_id = np.array([], dtype=int)
        return detections_all[detections_all.class_id != -1]


def initiate_polygon_zones(polygons, anchors=None):
    if anchors is None:
        anchors = [sv.Position.CENTER]
    return [sv.PolygonZone(polygon=p, triggering_anchors=anchors) for p in polygons]


# ============================================================
# 4) VIDEO PROCESSOR
# ============================================================
ZONE_IN_POLYGONS = [np.array([[100, 300], [200, 300], [200, 200], [100, 200]])]
ZONE_OUT_POLYGONS = [np.array([[100, 100], [200, 100], [200, 50], [100, 50]])]


class LaneVehicleProcessor:
    def __init__(self, weights, video, target=None, conf=0.3, iou=0.5, polygons=None):
        self.lane_detector = LaneDetector(
            json_path=polygons or DEFAULT_POLYGONS,
            video_path=video,
        )
        self.conf_threshold = conf
        self.iou_threshold = iou
        self.source_video_path = video
        self.target_video_path = target or DEFAULT_OUTPUT
        self.video_info = sv.VideoInfo.from_video_path(video)
        self.fps = self.video_info.fps

        # Modèle YOLOv8
        self.model = YOLO(weights)
        self.model.overrides.update({
            "conf": conf,
            "iou": iou,
            "device": "0" if _cuda() else "cpu",
            "half": _cuda(),
            "verbose": False,
        })

        # Tracker ByteTrack (paramètres recommandés)
        self.tracker = sv.ByteTrack(
            track_thresh=0.25,
            track_buffer=30,
            match_thresh=0.8,
        )

        # Annotateurs (initialisés 1 seule fois)
        self.colors = sv.ColorPalette.from_hex([
            "#FF0000", "#FFFF00", "#00FF00",
            "#800080", "#00FFFF", "#FFFFFF",
        ])
        self.box_annotator = sv.BoxAnnotator(color=self.colors, thickness=1)
        self.label_annotator = sv.LabelAnnotator(
            text_color=sv.Color.WHITE, text_padding=5, text_thickness=1
        )
        self.trace_annotator = sv.TraceAnnotator(
            color=self.colors, position=sv.Position.CENTER,
            trace_length=10, thickness=1,
        )

        # Filtre classes véhicules
        vehicles = {"car", "motorcycle", "bus", "truck", "bicycle"}
        if isinstance(self.model.names, dict):
            self.vehicle_class_ids = [
                i for i, n in self.model.names.items() if n in vehicles
            ]
        else:
            self.vehicle_class_ids = [
                i for i, n in enumerate(self.model.names) if n in vehicles
            ]

        # Zones de flux
        self.zones_in = initiate_polygon_zones(ZONE_IN_POLYGONS)
        self.zones_out = initiate_polygon_zones(ZONE_OUT_POLYGONS)
        self.detections_manager = DetectionsManager()

        # Suivi réel des vitesses
        self.previous_positions = {}
        self.speed_records = defaultdict(list)

    def _calculate_vehicle_speed(self, tracker_id, center):
        speed = 0.0
        if tracker_id in self.previous_positions:
            px, py = self.previous_positions[tracker_id]
            dist = math.hypot(center[0] - px, center[1] - py)
            raw = dist * self.fps
            if 0 < raw <= 1000:  # filtrage aberrations
                self.speed_records[tracker_id].append(raw)
                if len(self.speed_records[tracker_id]) > 30:
                    self.speed_records[tracker_id].pop(0)
        self.previous_positions[tracker_id] = center
        if self.speed_records[tracker_id]:
            return float(np.mean(self.speed_records[tracker_id]))
        return 0.0

    def process_video(self, display=False, target_fps=12):
        from tqdm import tqdm

        stride = max(1, int(self.video_info.fps / target_fps))
        gen = sv.get_video_frames_generator(self.source_video_path, stride=stride)

        if not self.target_video_path.lower().endswith(".mp4"):
            self.target_video_path += ".mp4"
        os.makedirs(os.path.dirname(self.target_video_path) or ".", exist_ok=True)

        total = int(self.video_info.total_frames / stride)
        processed = 0

        with sv.VideoSink(self.target_video_path, self.video_info) as sink:
            for frame in tqdm(gen, total=total, desc="🚀 Processing"):
                if frame.shape[1] > 1280:
                    frame = cv2.resize(frame, (1280, 720))
                out = self.process_frame(frame)
                sink.write_frame(out)
                processed += 1
                if display and processed % 5 == 0:
                    cv2.imshow("Preview", cv2.resize(out, (640, 360)))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        if display:
            cv2.destroyAllWindows()
        print(f"✅ Vidéo sauvegardée : {self.target_video_path}")

    def process_frame(self, frame):
        with _no_grad():
            results = self.model(
                frame, imgsz=640, augment=False,
                conf=self.conf_threshold, iou=self.iou_threshold,
                device="0" if _cuda() else "cpu", half=_cuda(),
            )[0]

        detections = sv.Detections.from_ultralytics(results)

        if len(detections) > 0:
            mask = np.logical_and(
                np.isin(detections.class_id, self.vehicle_class_ids),
                [self.lane_detector.is_in_detection_zone(b) for b in detections.xyxy],
            )
            detections = detections[mask]

        if len(detections) > 0:
            detections = self.tracker.update_with_detections(detections)
            original = sv.Detections(
                xyxy=detections.xyxy.copy(),
                confidence=detections.confidence,
                class_id=detections.class_id,
                tracker_id=detections.tracker_id,
            )
        else:
            original = sv.Detections.empty()

        det_in = [detections[z.trigger(detections=detections)] for z in self.zones_in]
        det_out = [detections[z.trigger(detections=detections)] for z in self.zones_out]

        # Comptage + vitesse réelle
        self.lane_detector.reset_counts()
        if len(original) > 0:
            centers = (original.xyxy[:, :2] + original.xyxy[:, 2:]) / 2
            for center, tid in zip(centers, original.tracker_id):
                lane = self.lane_detector.determine_lane((center[0], center[1]))
                self.lane_detector.count_vehicle(lane)
                spd = self._calculate_vehicle_speed(tid, (center[0], center[1]))
                self.lane_detector.add_vehicle_speed(lane, spd)

        labels = [
            self.model.names.get(int(c), "?") for c in original.class_id
        ] if len(original) > 0 else []

        flow = self.detections_manager.update(original, det_in, det_out) \
            if len(original) > 0 else sv.Detections.empty()

        return self.annotate_frame(frame, original, labels, flow)

    def annotate_frame(self, frame, detections, labels, flow):
        annotated = frame.copy()

        # Polygones des lanes
        for i, name in enumerate(self.lane_detector.lane_names):
            poly = self.lane_detector.lane_polygons[name]
            color = self.colors.colors[i % len(self.colors.colors)]
            annotated = sv.draw_filled_polygon(annotated, poly, color, opacity=0.3)
            annotated = sv.draw_polygon(annotated, poly, color, thickness=2)

        # Dashboard
        self._draw_dashboard(annotated)

        # Détections
        if len(detections) > 0:
            annotated = self.trace_annotator.annotate(annotated, detections)
            annotated = self.box_annotator.annotate(annotated, detections)
            custom = []
            for xyxy, lbl in zip(detections.xyxy, labels):
                x1, y1, x2, y2 = map(int, xyxy)
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                lane = self.lane_detector.determine_lane(center)
                custom.append(f"{lbl} ({lane})")
            annotated = self.label_annotator.annotate(
                annotated, detections, labels=custom
            )

        return annotated

    def _draw_dashboard(self, img):
        h, w = img.shape[:2]
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (420, 160), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

        total = sum(self.lane_detector.vehicle_counts.values())
        in_zone = sum(
            c for name, c in self.lane_detector.vehicle_counts.items()
            if name in ("emergency_lane", "middle_lane1", "middle_lane2")
        )
        speeds = [s for lst in self.lane_detector.lane_speeds.values() for s in lst]
        avg = round(float(np.mean(speeds)), 1) if speeds else 0.0
        mx = round(float(np.max(speeds)), 1) if speeds else 0.0

        cv2.putText(img, "Dashboard:", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"Total vehicles: {total}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f"Vehicules en zone: {in_zone}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"Vitesse moyenne: {avg} px/s", (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f"Vitesse max: {mx} px/s", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1, cv2.LINE_AA)

        # Lanes stats à droite
        y = 30
        for name in self.lane_detector.lane_names:
            cnt = self.lane_detector.vehicle_counts[name]
            sp = self.lane_detector.lane_average_speeds[name]
            txt = f"{name}: {cnt} veh. | {sp} px/s"
            cv2.putText(img, txt, (w - 400, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y += 22

        # Alert emergency lane
        emg = self.lane_detector.vehicle_counts.get("emergency_lane", 0)
        if emg > 0:
            cv2.putText(img, f"⚠ ALERT! {emg} vehicle(s) on emergency lane",
                        (w // 2 - 300, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)


# ============================================================
# HELPERS torch
# ============================================================
def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _no_grad():
    try:
        import torch
        return torch.no_grad()
    except ImportError:
        from contextlib import nullcontext
        return nullcontext()


# ============================================================
# 5) CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Vehicle detection, tracking and analytics (YOLOv8 + ByteTrack)"
    )
    p.add_argument("--video", required=True, help="Chemin de la vidéo d'entrée")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Poids YOLOv8 (ex: yolov8s.pt)")
    p.add_argument("--polygons", default=DEFAULT_POLYGONS, help="JSON des ROI/lanes")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="Chemin de la vidéo de sortie")
    p.add_argument("--conf", type=float, default=0.3, help="Seuil de confiance YOLO")
    p.add_argument("--iou", type=float, default=0.5, help="Seuil IoU YOLO")
    p.add_argument("--display", action="store_true", help="Afficher l'aperçu en direct")
    p.add_argument("--fps-target", type=int, default=12, help="FPS de traitement cible")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.video):
        raise FileNotFoundError(f"Vidéo introuvable : {args.video}")

    processor = LaneVehicleProcessor(
        weights=args.model,
        video=args.video,
        target=args.output,
        conf=args.conf,
        iou=args.iou,
        polygons=args.polygons,
    )
    processor.process_video(display=args.display, target_fps=args.fps_target)


if __name__ == "__main__":
    main()