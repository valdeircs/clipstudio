"""Local, static portrait framing from sampled faces. No network services."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

from PIL import Image

ZOOM_PRESETS = {'wide': 1.0, 'medium': 1.25, 'close': 1.5}
ZOOM_MODES = {'auto', 'manual', *ZOOM_PRESETS}

_VISION_SWIFT = r'''
import Foundation
import Vision
var outputs: [[String: Any]] = []
for path in CommandLine.arguments.dropFirst() {
    do {
        let request = VNDetectFaceRectanglesRequest()
        request.usesCPUOnly = true
        try VNImageRequestHandler(url: URL(fileURLWithPath: path), options: [:]).perform([request])
        let faces = (request.results ?? []).map { face -> [Double] in
            let b = face.boundingBox
            return [Double(b.minX), 1 - Double(b.maxY), Double(b.width), Double(b.height)]
        }
        outputs.append(["faces": faces])
    } catch {
        outputs.append(["faces": [], "unavailable": true])
    }
}
let data = try JSONSerialization.data(withJSONObject: outputs)
FileHandle.standardOutput.write(data)
'''


def _faces(paths: list[Path], cache: Path) -> tuple[list[list[list[float]]], str]:
    """Use an installed OpenCV detector or the built-in macOS Vision framework."""
    try:
        import cv2
    except ImportError:
        cv2 = None
    if cv2 is not None:
        classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        results = []
        for path in paths:
            frame = cv2.imread(str(path))
            height, width = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
            results.append([[float(x / width), float(y / height), float(w / width), float(h / height)] for x, y, w, h in found])
        return results, 'OpenCV'
    swift = shutil.which('swift')
    if sys.platform != 'darwin' or not swift:
        return [], 'unavailable'
    cache.mkdir(parents=True, exist_ok=True)
    script = cache / 'detect-faces.swift'
    if not script.exists() or script.read_text() != _VISION_SWIFT:
        script.write_text(_VISION_SWIFT)
    result = subprocess.run([swift, '-module-cache-path', str(cache / 'modules'), str(script), *map(str, paths)], capture_output=True, timeout=90, check=True)
    decoded = json.loads(result.stdout)
    if all(row.get('unavailable') for row in decoded):
        return [], 'unavailable'
    return [row.get('faces', []) for row in decoded], 'macOS Vision'


def source_crop_box(source_width: int, source_height: int, width: int, height: int,
                    zoom: float, position: float, vertical_position: float) -> tuple[float, float, float, float]:
    scale = max(width / source_width, height / source_height) * zoom
    crop_width, crop_height = width / scale, height / scale
    left = (source_width - crop_width) * position / 100
    top = (source_height - crop_height) * vertical_position / 100
    return left, top, left + crop_width, top + crop_height


def _from_faces(detections: list[list[list[float]]], source_width: int, source_height: int,
                fallback_position: float, fallback_vertical: float, detector: str) -> dict:
    base = {'zoom': 1.0, 'position': fallback_position, 'vertical_position': fallback_vertical,
            'fit': 'crop', 'detected_frames': 0, 'sampled_frames': len(detections), 'detector': detector}
    candidates = [[f for f in faces if len(f) == 4 and f[2] > .025 and f[3] > .025] for faces in detections]
    # Prefer one consistent subject across frames. Selecting the largest face in
    # each frame independently can jump to a face in a projected photograph.
    dominant, best_score = [], 0.0
    for faces in candidates:
        for anchor in faces:
            ax, ay = anchor[0] + anchor[2] / 2, anchor[1] + anchor[3] / 2
            track = []
            for frame_faces in candidates:
                nearby = [f for f in frame_faces if abs(f[0] + f[2] / 2 - ax) <= .12
                          and abs(f[1] + f[3] / 2 - ay) <= .15 and .5 <= f[3] / anchor[3] <= 2]
                if nearby:
                    track.append(min(nearby, key=lambda f: (f[0] + f[2] / 2 - ax) ** 2 + (f[1] + f[3] / 2 - ay) ** 2))
            area = statistics.median(f[2] * f[3] for f in track)
            score = len(track) * area * (1.5 - .5 * abs(ax - fallback_position / 100))
            if score > best_score:
                dominant, best_score = track, score
    if len(dominant) < 2:
        return {**base, 'reason': 'Automatic framing could not find a consistent face; using the widest crop at your selected position.'}
    aspect = source_width / source_height
    target = 9 / 16
    base_width, base_height = (target / aspect, 1.0) if aspect >= target else (1.0, aspect / target)
    # Include a generous head/shoulder margin across every sampled face position.
    left = max(0.0, min(x - w * .35 for x, y, w, h in dominant))
    right = min(1.0, max(x + w * 1.35 for x, y, w, h in dominant))
    top = max(0.0, min(y - h * .55 for x, y, w, h in dominant))
    bottom = min(1.0, max(y + h * 1.85 for x, y, w, h in dominant))
    safe_zoom = min(base_width / max(.001, right - left), base_height / max(.001, bottom - top))
    if safe_zoom < 1.0:
        return {**base, 'fit': 'blur', 'detected_frames': len(dominant),
                'reason': 'Detected faces move beyond a safe portrait crop; keeping the full video with a blurred background.'}
    zoom = max(1.0, min(1.5, .23 / statistics.median(f[3] for f in dominant), safe_zoom))
    crop_width, crop_height = base_width / zoom, base_height / zoom
    center_x = statistics.median(f[0] + f[2] / 2 for f in dominant)
    center_y = statistics.median(f[1] + f[3] / 2 for f in dominant)
    # Fit the union first, then place the face close to the upper third where possible.
    crop_left = max(max(0, right - crop_width), min(center_x - crop_width / 2, min(left, 1 - crop_width)))
    crop_top = max(max(0, bottom - crop_height), min(center_y - crop_height * .30, min(top, 1 - crop_height)))
    horizontal = crop_left / (1 - crop_width) * 100 if crop_width < .999999 else 50.0
    vertical = crop_top / (1 - crop_height) * 100 if crop_height < .999999 else 50.0
    return {**base, 'zoom': zoom, 'position': horizontal, 'vertical_position': vertical,
            'detected_frames': len(dominant),
            'reason': f'Automatic framing selected one static crop from faces found in {len(dominant)} of {len(detections)} sampled frames.'}


def suggest_framing(source: Path, start: float, end: float, ffmpeg: str, work: Path,
                    fallback_position: float = 50, fallback_vertical: float = 50) -> dict:
    fallback = {'zoom': 1.0, 'position': fallback_position, 'vertical_position': fallback_vertical,
                'fit': 'crop', 'detected_frames': 0, 'sampled_frames': 0, 'detector': 'unavailable',
                'reason': 'Automatic face detection is unavailable; using the widest crop at your selected position.'}
    try:
        frames = []
        for index, fraction in enumerate((.15, .50, .85)):
            frame = work / f'framing-{index}.jpg'
            subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-ss', str(start + (end - start) * fraction),
                            '-i', str(source), '-frames:v', '1', '-vf', 'scale=640:-2', str(frame)],
                           capture_output=True, timeout=30, check=True)
            frames.append(frame)
        with Image.open(frames[0]) as image:
            width, height = image.size
        # Cache SDK compilation outside transient render folders; no binary ships with the app.
        cache = Path(__file__).resolve().parent / 'data' / 'framing-helper'
        detections, detector = _faces(frames, cache)
        if detector == 'unavailable':
            return fallback
        return _from_faces(detections, width, height, fallback_position, fallback_vertical, detector)
    except (OSError, subprocess.SubprocessError, ValueError, TypeError, KeyError, AttributeError):
        return fallback
