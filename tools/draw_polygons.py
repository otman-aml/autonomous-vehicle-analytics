"""
Outil interactif pour dessiner les polygones de ROI (Region of Interest)
sur une frame extraite d'une vidéo.

Permet de générer un fichier `polygons.json` compatible avec le pipeline
`vehicle_analytics.py` (détection + tracking + analyse).

Usage:
    python tools/draw_polygons.py --video data/demo_short.mp4 --output data/polygons.json

Contrôles:
    - Clic gauche   : ajouter un sommet au polygone courant
    - ESPACE        : valider le polygone courant
    - 'u'           : annuler le dernier point
    - 'r'           : réinitialiser tous les polygones
    - ESC           : terminer et sauvegarder le JSON

Labels prédéfinis (dans l'ordre):
    emniyet_seridi → sag_serit → orta_serit1 → orta_serit2 → sol_serit

Auteur : Otman Amal — M2 SIIA, FPK Khouribga
"""

import argparse
import json
import os

import cv2
import numpy as np

# Labels par défaut (traduits automatiquement par vehicle_analytics.py)
DEFAULT_LABELS = [
    "emniyet_seridi",   # voie d'urgence
    "sag_serit",        # voie de droite
    "orta_serit1",      # voie du milieu 1
    "orta_serit2",      # voie du milieu 2
    "sol_serit",        # voie de gauche
]

# Couleurs BGR pour l'affichage
COLORS = [
    (0, 0, 255),      # rouge
    (0, 255, 255),    # jaune
    (0, 255, 0),      # vert
    (128, 0, 128),    # violet
    (255, 255, 0),    # cyan
]


class PolygonDrawer:
    def __init__(self, frame: np.ndarray, labels: list):
        self.frame = frame
        self.labels = labels
        self.polygons = []       # list[list[[x, y]]]
        self.current = []        # points du polygone en cours
        self.window = "Draw ROI — clic gauche: point | ESPACE: valider | u: undo | r: reset | ESC: fin"

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current.append([float(x), float(y)])

    def _render(self) -> np.ndarray:
        img = self.frame.copy()

        # Polygones validés
        for i, poly in enumerate(self.polygons):
            color = COLORS[i % len(COLORS)]
            pts = np.array(poly, dtype=np.int32)
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], color)
            img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

            # Label au centroïde
            cx = int(np.mean([p[0] for p in poly]))
            cy = int(np.mean([p[1] for p in poly]))
            cv2.putText(img, self.labels[i], (cx - 40, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Polygone en cours
        if self.current:
            pts = np.array(self.current, dtype=np.int32)
            cv2.polylines(img, [pts], isClosed=False, color=(0, 0, 255), thickness=2)
            for p in self.current:
                cv2.circle(img, (int(p[0]), int(p[1])), 5, (0, 0, 255), -1)

        # Barre d'état
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w, 50), (0, 0, 0), -1)
        next_label = self.labels[len(self.polygons)] if len(self.polygons) < len(self.labels) else "extra"
        status = (
            f"Polygone {len(self.polygons) + 1}/{len(self.labels)} — "
            f"label: {next_label} | points: {len(self.current)}"
        )
        cv2.putText(img, status, (15, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2, cv2.LINE_AA)
        return img

    def run(self) -> list:
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self._on_mouse)

        while True:
            cv2.imshow(self.window, self._render())
            k = cv2.waitKey(20) & 0xFF

            if k == 32:  # ESPACE
                if len(self.current) >= 3:
                    self.polygons.append(self.current.copy())
                    self.current.clear()
                else:
                    print("⚠️  Un polygone doit avoir au moins 3 points.")

            elif k == ord("u"):  # undo
                if self.current:
                    self.current.pop()

            elif k == ord("r"):  # reset
                self.polygons.clear()
                self.current.clear()

            elif k == 27:  # ESC
                break

        cv2.destroyAllWindows()
        return self.polygons


def build_json(polygons: list, labels: list, width: int, height: int) -> dict:
    """Construit le dict JSON au format attendu par vehicle_analytics.py."""
    boxes = []
    for i, poly in enumerate(polygons):
        label = labels[i] if i < len(labels) else f"lane_{i + 1}"
        boxes.append({
            "id": str(i + 1),
            "label": label,
            "points": poly,
        })

    # ROI globale (id=6) — utile comme zone de détection fallback
    boxes.append({
        "id": "6",
        "label": "emniyet_seridi",
        "points": [[0, 0], [width, 0], [width, height], [0, height]],
    })

    return {
        "width": width,
        "height": height,
        "boxes": boxes,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Dessine les polygones de ROI et génère polygons.json"
    )
    parser.add_argument("--video", required=True,
                        help="Chemin de la vidéo (utilise la 1ère frame)")
    parser.add_argument("--output", default="data/polygons.json",
                        help="Chemin du fichier JSON de sortie")
    parser.add_argument("--frame", type=int, default=0,
                        help="Index de la frame à utiliser (default: 0)")
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        raise FileNotFoundError(f"Vidéo introuvable: {args.video}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo: {args.video}")

    if args.frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)

    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Impossible de lire la frame demandée")

    h, w = frame.shape[:2]
    print(f"📐 Résolution: {w}×{h}")
    print(f"🖱️  Dessinez jusqu'à {len(DEFAULT_LABELS)} polygones (un par voie).")
    print("   ESPACE = valider | 'u' = undo | 'r' = reset | ESC = terminer")

    drawer = PolygonDrawer(frame, DEFAULT_LABELS)
    polygons = drawer.run()

    if not polygons:
        print("⚠️  Aucun polygone dessiné — fichier non généré.")
        return

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    data = build_json(polygons, DEFAULT_LABELS, w, h)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ {len(polygons)} polygones sauvegardés dans {args.output}")


if __name__ == "__main__":
    main()