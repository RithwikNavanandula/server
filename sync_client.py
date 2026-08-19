"""
Push annotated frames + detection events from Windows edge → PythonAnywhere cloud.

Env (set in .env next to this file, or system env):
  CLOUD_URL          e.g. https://youruser.pythonanywhere.com
  EDGE_SYNC_SECRET   must match cloud Config.EDGE_SYNC_SECRET
  EDGE_FRAME_FPS     default 3 (PA relay rate)
"""
from __future__ import annotations

import base64
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load .env next to edge app
_env = Path(__file__).resolve().parent / '.env'
if _env.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env)
    except ImportError:
        for line in _env.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, _, v = line.partition('=')
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

import requests

logger = logging.getLogger('ai_cctv.edge_sync')

CLOUD_URL = os.getenv('CLOUD_URL', '').rstrip('/')
EDGE_SYNC_SECRET = os.getenv('EDGE_SYNC_SECRET', 'change-me-edge-sync')
FRAME_INTERVAL = 1.0 / max(0.5, float(os.getenv('EDGE_FRAME_FPS', '3')))


class CloudSync:
    """Background pusher: frames (JPEG) + batched detections/session to cloud."""

    def __init__(self):
        self._det_q: queue.Queue = queue.Queue(maxsize=500)
        self._running = False
        self._threads: List[threading.Thread] = []
        self._camera_getter = None  # callable -> VideoCamera | None
        self._last_session = {}

    def configure(self, camera_getter):
        self._camera_getter = camera_getter

    @property
    def enabled(self) -> bool:
        return bool(CLOUD_URL)

    def enqueue_detection(self, detection_type: str, confidence: float,
                          direction: Optional[str], inference_ms: float = 0,
                          count: int = 1):
        if not self.enabled:
            return
        try:
            self._det_q.put_nowait({
                'type': detection_type,
                'confidence': confidence,
                'direction': direction,
                'inference_ms': inference_ms,
                'count': count,
            })
        except queue.Full:
            logger.warning('edge sync detection queue full — dropping event')

    def start(self):
        if not self.enabled:
            logger.warning('CLOUD_URL not set — edge→cloud sync disabled')
            return
        if self._running:
            return
        self._running = True
        t1 = threading.Thread(target=self._frame_loop, daemon=True, name='edge-frame-sync')
        t2 = threading.Thread(target=self._event_loop, daemon=True, name='edge-event-sync')
        self._threads = [t1, t2]
        for t in self._threads:
            t.start()
        logger.info(f'☁️ Edge sync started → {CLOUD_URL} (frame every {FRAME_INTERVAL:.2f}s)')

    def stop(self):
        self._running = False

    def _headers(self) -> Dict[str, str]:
        return {'X-Edge-Secret': EDGE_SYNC_SECRET}

    def _frame_loop(self):
        while self._running:
            try:
                cam = self._camera_getter() if self._camera_getter else None
                jpeg = None
                session = {}
                if cam is not None:
                    if hasattr(cam, 'get_encoded_frame'):
                        jpeg = cam.get_encoded_frame()
                    if jpeg is None and hasattr(cam, 'get_frame'):
                        frame = cam.get_frame()
                        if frame is not None:
                            import cv2
                            h, w = frame.shape[:2]
                            max_w = 640
                            if w > max_w:
                                scale = max_w / w
                                frame = cv2.resize(frame, (max_w, int(h * scale)))
                            ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
                            if ok:
                                jpeg = buf.tobytes()
                    session = {
                        'session_in': getattr(cam, 'session_in', 0),
                        'session_out': getattr(cam, 'session_out', 0),
                        'line_position': getattr(cam, 'line_position', 0.45),
                        'confidence': getattr(cam, 'confidence', 0.5),
                        'frame_skip': getattr(cam, 'frame_skip', 3),
                    }
                    self._last_session = session

                if jpeg:
                    # Prefer multipart (smaller overhead than giant JSON)
                    try:
                        r = requests.post(
                            f'{CLOUD_URL}/api/edge/frame',
                            headers=self._headers(),
                            files={'frame': ('frame.jpg', jpeg, 'image/jpeg')},
                            timeout=8,
                        )
                        if r.status_code == 401:
                            logger.error('edge frame sync unauthorized — check EDGE_SYNC_SECRET')
                        elif r.status_code >= 400:
                            logger.warning(f'frame sync HTTP {r.status_code}: {r.text[:200]}')
                    except Exception as e:
                        logger.warning(f'frame sync failed: {e}')
            except Exception as e:
                logger.warning(f'frame loop error: {e}')
            time.sleep(FRAME_INTERVAL)

    def _event_loop(self):
        while self._running:
            batch: List[Dict[str, Any]] = []
            try:
                item = self._det_q.get(timeout=2.0)
                batch.append(item)
                while len(batch) < 50:
                    try:
                        batch.append(self._det_q.get_nowait())
                    except queue.Empty:
                        break
            except queue.Empty:
                # still heartbeat session periodically
                if self._last_session:
                    self._post_sync([], self._last_session)
                continue

            self._post_sync(batch, self._last_session)

    def _post_sync(self, detections: List[dict], session: dict):
        if not detections and not session:
            return
        try:
            r = requests.post(
                f'{CLOUD_URL}/api/edge/sync',
                headers={**self._headers(), 'Content-Type': 'application/json'},
                json={'detections': detections, 'session': session},
                timeout=8,
            )
            if r.status_code >= 400:
                logger.warning(f'event sync HTTP {r.status_code}: {r.text[:200]}')
        except Exception as e:
            logger.warning(f'event sync failed: {e}')


cloud_sync = CloudSync()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print('CloudSync module — import from edge app; set CLOUD_URL to enable.')
