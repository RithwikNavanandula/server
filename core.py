#!/usr/bin/env python3
"""
AI CCTV Backend - v4.1
Complete implementation with real stats, analytics, detection saving, inventory updates
"""

import os
import cv2
import sqlite3
import uuid
import subprocess
import numpy as np
import threading
import time
import jwt
import logging
import re
import csv
import io
import json
from datetime import datetime, timedelta
from pathlib import Path
from functools import wraps
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Generator
from enum import Enum
from urllib.parse import urlparse

from flask import Flask, jsonify, request, Response, send_file, send_from_directory, g
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

# Path to the built React frontend


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    BASE_DIR = Path(__file__).parent
    MODEL_DIR = Path(os.getenv('MODEL_DIR', str(BASE_DIR / 'models')))
    UPLOAD_DIR = Path(os.getenv('UPLOAD_DIR', str(BASE_DIR / 'uploads')))
    DB_PATH = os.getenv('DB_PATH', str(BASE_DIR / 'aicctv.db'))
    JWT_SECRET = os.getenv('JWT_SECRET', 'ai-cctv-secret-key-2024')
    JWT_EXPIRY_HOURS = int(os.getenv('JWT_EXPIRY_HOURS', 24))
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*')
    RATE_LIMIT_REQUESTS = int(os.getenv('RATE_LIMIT_REQUESTS', 100))
    RATE_LIMIT_WINDOW = int(os.getenv('RATE_LIMIT_WINDOW', 60))
    ML_DEVICE = os.getenv('ML_DEVICE', 'cpu')  # ProLiant has no GPU
    ML_DEFAULT_CONFIDENCE = float(os.getenv('ML_CONFIDENCE', 0.5))
    ML_MAX_DETECTIONS = int(os.getenv('ML_MAX_DETECTIONS', 50))
    COMPRESSION_TIMEOUT = int(os.getenv('COMPRESSION_TIMEOUT', 600))
    MAX_UPLOAD_SIZE = int(os.getenv('MAX_UPLOAD_MB', 500)) * 1024 * 1024
    FRONTEND_DIST = Path(os.getenv('FRONTEND_DIST', str(BASE_DIR / 'static')))
    LIVE_RECORDING_DIR = Path(os.getenv('LIVE_RECORDING_DIR', str(BASE_DIR / 'recordings')))
    LIVE_RECORDING_FPS = float(os.getenv('LIVE_RECORDING_FPS', 2))
    LIVE_RECORDING_CHUNK_SECONDS = int(os.getenv('LIVE_RECORDING_CHUNK_SECONDS', 300))
    LIVE_RECORDING_RETENTION_HOURS = int(os.getenv('LIVE_RECORDING_RETENTION_HOURS', 12))

Config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
Config.LIVE_RECORDING_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('ai_cctv')

# ============================================================================
# ML INITIALIZATION
# ============================================================================

try:
    os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'
    import torch
    DEVICE = 'cuda' if (Config.ML_DEVICE == 'auto' and torch.cuda.is_available()) else Config.ML_DEVICE if Config.ML_DEVICE != 'auto' else 'cpu'
    if DEVICE == 'cuda':
        logger.info(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("💻 Running on CPU")
except ImportError:
    torch = None
    DEVICE = 'cpu'
    logger.warning("⚠️ PyTorch not available")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    logger.info("✅ YOLO available")
except ImportError:
    YOLO = None
    YOLO_AVAILABLE = False
    logger.warning("⚠️ YOLO not available")

# Optional barcode support
try:
    from pyzbar import pyzbar
    BARCODE_AVAILABLE = True
except ImportError:
    BARCODE_AVAILABLE = False

# ============================================================================
# EXCEPTIONS
# ============================================================================

class APIError(Exception):
    def __init__(self, message: str, status_code: int = 400, details: dict = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details
    def to_dict(self) -> dict:
        result = {'error': self.message, 'status_code': self.status_code}
        if self.details: result['details'] = self.details
        return result

class ValidationError(APIError):
    def __init__(self, message: str, errors: dict = None): super().__init__(message, 400, errors)

class AuthenticationError(APIError):
    def __init__(self, message: str = "Authentication required"): super().__init__(message, 401)

class AuthorizationError(APIError):
    def __init__(self, message: str = "Permission denied"): super().__init__(message, 403)

class NotFoundError(APIError):
    def __init__(self, resource: str = "Resource"): super().__init__(f"{resource} not found", 404)

# ============================================================================
# VALIDATION
# ============================================================================

class Validator:
    def __init__(self, data: dict):
        self.data = data or {}
        self.errors: Dict[str, List[str]] = {}
    
    def require(self, field: str, message: str = None) -> 'Validator':
        value = self.data.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            self._add_error(field, message or f"{field} is required")
        return self
    
    def email(self, field: str) -> 'Validator':
        value = self.data.get(field, '')
        if value and not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', value):
            self._add_error(field, "Invalid email format")
        return self
    
    def min_length(self, field: str, length: int) -> 'Validator':
        value = self.data.get(field, '')
        if value and len(str(value)) < length:
            self._add_error(field, f"{field} must be at least {length} characters")
        return self
    
    def max_length(self, field: str, length: int) -> 'Validator':
        value = self.data.get(field, '')
        if value and len(str(value)) > length:
            self._add_error(field, f"{field} must be at most {length} characters")
        return self
    
    def in_list(self, field: str, allowed: list) -> 'Validator':
        value = self.data.get(field)
        if value is not None and value not in allowed:
            self._add_error(field, f"{field} must be one of: {', '.join(map(str, allowed))}")
        return self
    
    def _add_error(self, field: str, message: str) -> None:
        if field not in self.errors: self.errors[field] = []
        self.errors[field].append(message)
    
    def validate(self) -> dict:
        if self.errors: raise ValidationError("Validation failed", self.errors)
        return self.data

# ============================================================================
# RATE LIMITER
# ============================================================================

class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
    
    def is_allowed(self, key: str) -> Tuple[bool, int]:
        now = time.time()
        with self._lock:
            if key not in self.requests: self.requests[key] = []
            self.requests[key] = [t for t in self.requests[key] if now - t < self.window]
            remaining = self.max_requests - len(self.requests[key])
            if len(self.requests[key]) >= self.max_requests: return False, 0
            self.requests[key].append(now)
            return True, remaining - 1
    
    def get_key(self) -> str:
        return getattr(request, 'user_id', None) or request.remote_addr or 'anonymous'
    
    def cleanup(self) -> None:
        """Remove stale keys to prevent unbounded memory growth."""
        now = time.time()
        with self._lock:
            stale = [k for k, ts in self.requests.items()
                     if not ts or all(now - t >= self.window for t in ts)]
            for k in stale:
                del self.requests[k]

rate_limiter = RateLimiter(Config.RATE_LIMIT_REQUESTS, Config.RATE_LIMIT_WINDOW)

# ============================================================================
# DATABASE
# ============================================================================

class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()
    
    @property
    def connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
            self._local.connection.row_factory = sqlite3.Row
            self._local.connection.execute("PRAGMA foreign_keys = ON")
        return self._local.connection
    
    def close(self) -> None:
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
    
    @contextmanager
    def get_cursor(self):
        cursor = self.connection.cursor()
        try:
            yield cursor
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        finally:
            cursor.close()
    
    def execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        return self.connection.execute(query, params)
    
    def fetchone(self, query: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        return self.execute(query, params).fetchone()
    
    def fetchall(self, query: str, params: tuple = ()) -> List[sqlite3.Row]:
        return self.execute(query, params).fetchall()
    
    def insert(self, table: str, data: dict) -> str:
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        with self.get_cursor() as cursor:
            cursor.execute(query, tuple(data.values()))
            return data.get('id', cursor.lastrowid)
    
    def update(self, table: str, data: dict, where: str, where_params: tuple) -> int:
        set_clause = ', '.join([f"{k} = ?" for k in data.keys()])
        query = f"UPDATE {table} SET {set_clause} WHERE {where}"
        with self.get_cursor() as cursor:
            cursor.execute(query, tuple(data.values()) + where_params)
            return cursor.rowcount
    
    def delete(self, table: str, where: str = "1=1", params: tuple = ()) -> int:
        with self.get_cursor() as cursor:
            cursor.execute(f"DELETE FROM {table} WHERE {where}", params)
            return cursor.rowcount
    
    def init_schema(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            name TEXT NOT NULL,
            role TEXT DEFAULT 'viewer',
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS detections (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            confidence REAL NOT NULL,
            direction TEXT,
            inference_ms REAL,
            camera_id TEXT,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS inventory (
            id TEXT PRIMARY KEY,
            product_name TEXT UNIQUE NOT NULL,
            count_in INTEGER DEFAULT 0,
            count_out INTEGER DEFAULT 0,
            current_stock INTEGER DEFAULT 0,
            min_threshold INTEGER DEFAULT 10,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS trucks (
            id TEXT PRIMARY KEY,
            plate_number TEXT NOT NULL,
            direction TEXT DEFAULT 'IN',
            driver_name TEXT,
            company TEXT,
            purpose TEXT,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            exit_at TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS scans (
            id TEXT PRIMARY KEY,
            barcode TEXT,
            product_name TEXT,
            batch_number TEXT,
            mfg_date TEXT,
            exp_date TEXT,
            rack_no TEXT,
            shelf_no TEXT,
            direction TEXT DEFAULT 'IN',
            scanned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            encoding BLOB,
            image_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS alerts (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            message TEXT NOT NULL,
            severity TEXT DEFAULT 'info',
            is_read INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- ── Phase-1 supply-chain tables ─────────────────────────────
        CREATE TABLE IF NOT EXISTS bays (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            camera_source TEXT DEFAULT '0',
            location TEXT,
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS shifts (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ended_at TIMESTAMP,
            operator_name TEXT,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS loading_jobs (
            id TEXT PRIMARY KEY,
            bay_id TEXT,
            truck_id TEXT,
            truck_plate TEXT NOT NULL,
            product_name TEXT DEFAULT 'Sugar Bag',
            target_count INTEGER NOT NULL,
            loaded_count INTEGER DEFAULT 0,
            direction TEXT DEFAULT 'OUT',
            shift_id TEXT,
            status TEXT DEFAULT 'pending',
            operator_name TEXT,
            notes TEXT,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (bay_id) REFERENCES bays(id),
            FOREIGN KEY (shift_id) REFERENCES shifts(id)
        );

        -- Extend detections to carry job/shift context
        -- (ALTER TABLE is idempotent via the try in seed_data)

        CREATE INDEX IF NOT EXISTS idx_detections_date ON detections(detected_at);
        CREATE INDEX IF NOT EXISTS idx_detections_type ON detections(type);
        CREATE INDEX IF NOT EXISTS idx_detections_direction ON detections(direction);
        CREATE INDEX IF NOT EXISTS idx_trucks_date ON trucks(detected_at);
        CREATE INDEX IF NOT EXISTS idx_trucks_plate ON trucks(plate_number);
        CREATE INDEX IF NOT EXISTS idx_scans_date ON scans(scanned_at);
        CREATE INDEX IF NOT EXISTS idx_scans_barcode ON scans(barcode);
        CREATE INDEX IF NOT EXISTS idx_alerts_read ON alerts(is_read);
        CREATE INDEX IF NOT EXISTS idx_jobs_status ON loading_jobs(status);
        CREATE INDEX IF NOT EXISTS idx_jobs_bay ON loading_jobs(bay_id);
        CREATE INDEX IF NOT EXISTS idx_shifts_date ON shifts(started_at);
        """
        with self.get_cursor() as cursor:
            cursor.executescript(schema)
        # Add columns that may not exist in older DBs (safe migration)
        for col_sql in [
            "ALTER TABLE detections ADD COLUMN shift_id TEXT",
            "ALTER TABLE detections ADD COLUMN job_id TEXT",
            "ALTER TABLE detections ADD COLUMN bay_id TEXT",
        ]:
            try:
                with self.get_cursor() as c:
                    c.execute(col_sql)
            except Exception:
                pass  # column already exists
        logger.info("✅ Database schema initialized")

    
    def seed_data(self) -> None:
        if not self.fetchone("SELECT id FROM users WHERE email = ?", ('demo@aicctv.com',)):
            self.insert('users', {
                'id': str(uuid.uuid4()),
                'email': 'demo@aicctv.com',
                'password_hash': generate_password_hash('demo123'),
                'name': 'Demo User',
                'role': 'admin'
            })
            logger.info("✅ Demo user created (demo@aicctv.com / demo123)")
        
        for product in ['Sugar Bag', 'Full Crate', 'Half Crate', 'Box', 'Pallet']:
            if not self.fetchone("SELECT id FROM inventory WHERE product_name = ?", (product,)):
                self.insert('inventory', {'id': str(uuid.uuid4()), 'product_name': product})
        logger.info("✅ Seed data loaded")

db = Database(Config.DB_PATH)

# ============================================================================
# HELPERS
# ============================================================================

def create_alert(alert_type: str, message: str, severity: str = 'info'):
    """Create a new alert"""
    db.insert('alerts', {
        'id': str(uuid.uuid4()),
        'type': alert_type,
        'message': message,
        'severity': severity
    })

def save_detection_to_db(detection_type: str, confidence: float, direction: str = None, inference_ms: float = 0):
    """Save a detection to local edge DB and queue for cloud sync."""
    try:
        db.insert('detections', {
            'id': str(uuid.uuid4()),
            'type': detection_type,
            'confidence': confidence,
            'direction': direction,
            'inference_ms': inference_ms
        })
        try:
            from sync_client import cloud_sync
            cloud_sync.enqueue_detection(detection_type, confidence, direction, inference_ms)
        except Exception:
            pass
    except Exception as e:
        logger.error(f"Failed to save detection: {e}")

def update_inventory_from_detection(product_type: str, direction: str = 'IN', count: int = 1):
    """Update inventory based on detection"""
    label_to_product = {
        'sugar_bag': 'Sugar Bag', 'sugar': 'Sugar Bag',
        'full_crate': 'Full Crate', 'crate': 'Full Crate',
        'half_crate': 'Half Crate', 'box': 'Box', 'pallet': 'Pallet'
    }
    product_name = label_to_product.get(product_type.lower(), product_type)
    
    item = db.fetchone("SELECT * FROM inventory WHERE LOWER(product_name) = LOWER(?)", (product_name,))
    
    if not item:
        db.insert('inventory', {
            'id': str(uuid.uuid4()),
            'product_name': product_name,
            'count_in': count if direction == 'IN' else 0,
            'count_out': count if direction == 'OUT' else 0,
            'current_stock': count if direction == 'IN' else 0
        })
    else:
        with db.get_cursor() as cursor:
            if direction == 'IN':
                cursor.execute("UPDATE inventory SET count_in = count_in + ?, current_stock = current_stock + ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (count, count, item['id']))
            else:
                cursor.execute("UPDATE inventory SET count_out = count_out + ?, current_stock = MAX(0, current_stock - ?), updated_at = CURRENT_TIMESTAMP WHERE id = ?", (count, count, item['id']))

def check_low_stock_alerts():
    """Check inventory for low stock and create alerts.
    Uses 2 queries total instead of N+1 LIKE scans.
    """
    low_items = db.fetchall("SELECT product_name, current_stock, min_threshold FROM inventory WHERE current_stock < min_threshold")
    if not low_items:
        return
    today = datetime.now().strftime('%Y-%m-%d')
    # Single query for all low_stock messages today; check membership in Python
    existing_rows = db.fetchall("SELECT message FROM alerts WHERE type = 'low_stock' AND DATE(created_at) = ?", (today,))
    alerted_today = {row['message'] for row in existing_rows}
    for item in low_items:
        msg = f"Low stock: {item['product_name']} has only {item['current_stock']} units (threshold: {item['min_threshold']})"
        if msg not in alerted_today:
            create_alert('low_stock', msg, 'warning')

# ============================================================================
# AUTHENTICATION
# ============================================================================

def create_token(user_id: str, role: str, name: str) -> str:
    return jwt.encode({'user_id': user_id, 'role': role, 'name': name, 'exp': datetime.utcnow() + timedelta(hours=Config.JWT_EXPIRY_HOURS)}, Config.JWT_SECRET, algorithm='HS256')

def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, Config.JWT_SECRET, algorithms=['HS256'])
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Token expired")
    except jwt.InvalidTokenError:
        raise AuthenticationError("Invalid token")

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            try:
                payload = decode_token(auth_header[7:])
                request.user_id = payload['user_id']
                request.user_role = payload['role']
                request.user_name = payload.get('name', '')
            except:
                request.user_id = 'demo'
                request.user_role = 'admin'
                request.user_name = 'Demo'
        else:
            request.user_id = 'demo'
            request.user_role = 'admin'
            request.user_name = 'Demo'
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            try:
                payload = decode_token(auth_header[7:])
                if payload.get('role') != 'admin': raise AuthorizationError()
                request.user_id = payload['user_id']
                request.user_role = payload['role']
            except AuthorizationError:
                raise
            except:
                request.user_id = 'demo'
                request.user_role = 'admin'
        else:
            request.user_id = 'demo'
            request.user_role = 'admin'
        return f(*args, **kwargs)
    return decorated

def rate_limit(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        allowed, remaining = rate_limiter.is_allowed(rate_limiter.get_key())
        if not allowed: return jsonify({'error': 'Rate limit exceeded'}), 429
        return f(*args, **kwargs)
    return decorated

# ============================================================================
# ML SERVICE
# ============================================================================

@dataclass
class DetectionResult:
    label: str
    confidence: float
    bbox: List[int]
    inference_ms: float
    
    def to_dict(self) -> dict:
        return {'label': self.label, 'confidence': round(self.confidence, 3), 'bbox': self.bbox, 'inference_ms': round(self.inference_ms, 1)}

class MLService:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized: return
        # Registry of discovered model paths — does NOT hold loaded model objects
        self._registry: Dict[str, Path] = {}
        # Only the currently active model is loaded into memory
        self._loaded_model: Any = None
        self._active_model: Optional[str] = None
        self._latest_detections: List[DetectionResult] = []
        self._lock = threading.Lock()
        self._initialized = True
    
    @property
    def device(self) -> str: return DEVICE
    @property
    def active_model_name(self) -> Optional[str]: return self._active_model
    @property
    def loaded_models(self) -> Dict[str, dict]:
        """Returns all discovered models; marks which one is currently loaded."""
        return {
            name: {'loaded': name == self._active_model, 'device': DEVICE if name == self._active_model else None}
            for name in self._registry
        }
    @property
    def latest_detections(self) -> List[dict]: return [d.to_dict() for d in self._latest_detections]
    
    def register_model(self, path: Path, name: str = None) -> bool:
        """Register a model path without loading it into memory."""
        name = name or path.stem
        if not path.exists():
            logger.warning(f"⚠️ Model file not found: {path}")
            return False
        self._registry[name] = path
        logger.info(f"📋 Model '{name}' registered (not yet loaded)")
        return True
    
    def register_models_from_directory(self, directory: Path) -> int:
        """Scan a directory and register all .pt files without loading them."""
        if not directory.exists(): return 0
        count = 0
        for mf in sorted(directory.glob('*.pt')):
            if self.register_model(mf): count += 1
        return count
    
    def _load_model_into_memory(self, name: str) -> bool:
        """Internal: actually load a registered model into memory."""
        if not YOLO_AVAILABLE: return False
        path = self._registry.get(name)
        if path is None:
            logger.error(f"❌ Model '{name}' not in registry")
            return False
        try:
            model = YOLO(str(path))
            if DEVICE == 'cuda': model.to('cuda')
            self._loaded_model = model
            logger.info(f"✅ Model '{name}' loaded into memory on {DEVICE}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load '{name}': {e}")
            return False
    
    def _unload_current_model(self):
        """Unload the current model from memory to free RAM."""
        if self._loaded_model is not None:
            logger.info(f"🗑️ Unloading model '{self._active_model}' from memory")
            self._loaded_model = None
            if torch is not None and DEVICE == 'cuda':
                torch.cuda.empty_cache()
    
    def switch_model(self, name: str) -> bool:
        """Switch active model: unloads the current one, loads the new one."""
        if name not in self._registry:
            return False
        if name == self._active_model:
            return True  # Already active
        with self._lock:
            self._unload_current_model()
            if self._load_model_into_memory(name):
                self._active_model = name
                return True
            return False
    
    # Legacy compatibility: kept so existing call-sites don't break
    def load_model(self, path: Path, name: str = None) -> bool:
        """Register and immediately load a model (used for the initial default)."""
        name = name or path.stem
        self.register_model(path, name)
        return self.switch_model(name)
    
    def load_models_from_directory(self, directory: Path) -> int:
        """Register all models in a directory (lazy — none are loaded yet)."""
        return self.register_models_from_directory(directory)
    
    def get_active_model(self) -> Any:
        return self._loaded_model
    
    def detect(self, frame: np.ndarray, confidence: float = 0.5, draw: bool = True) -> Tuple[np.ndarray, List[DetectionResult]]:
        model = self.get_active_model()
        if model is None: return frame, []
        
        with self._lock:
            start = time.time()
            results = model(frame, conf=confidence, verbose=False)
            ms = (time.time() - start) * 1000
            
            detections = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    label = model.names[int(box.cls[0])]
                    detections.append(DetectionResult(label, conf, [x1, y1, x2, y2], ms))
                    if draw:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            self._latest_detections = detections[:Config.ML_MAX_DETECTIONS]
            return frame, detections
    
    def health_check(self) -> dict:
        return {'device': DEVICE, 'models_loaded': 1 if self._loaded_model else 0, 'active_model': self._active_model}

ml_service = MLService()

# ============================================================================
# CAMERA & COMPRESSION
# ============================================================================

def normalize_camera_source(source: Any = 0) -> Any:
    if isinstance(source, str):
        raw = source.strip()
        if not raw:
            return 0
        try:
            return int(raw)
        except ValueError:
            return raw
    try:
        return int(source)
    except (TypeError, ValueError):
        return 0

def camera_source_kind(source: Any) -> str:
    if isinstance(source, int):
        return 'webcam'
    parsed = urlparse(str(source))
    return parsed.scheme.lower() or 'stream'

def camera_source_key(source: Any, camera_id: str = None) -> str:
    raw = camera_id or str(source)
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', raw).strip('_')[:80] or 'camera'

class VideoCamera:
    def __init__(self, source: Any = 0, camera_id: str = None):
        source = normalize_camera_source(source)
        self.source = source
        self.camera_id = camera_id or camera_source_key(source)
        self.source_kind = camera_source_kind(source)
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera source: {source}")
        self._raw_frame = None
        self._processed_frame = None
        self._frame_lock = threading.Lock()
        self._running = True

        # Tracker and positions
        self.tracker = SimpleCentroidTracker()
        self.previous_positions = {}
        self.line_position = 0.45  # 45% of width — configurable via PATCH /api/camera/line

        # Live-adjustable inference settings (no restart needed)
        self.confidence = Config.ML_DEFAULT_CONFIDENCE
        self.frame_skip = 3  # process every Nth frame
        self._frame_counter = 0

        # Live session crossing counters (reset when camera restarts)
        self.session_in = 0
        self.session_out = 0

        # Rolling raw-video recording with metadata for on-demand annotated export.
        self.recording_enabled = os.getenv('LIVE_RECORDING_ENABLED', 'true').lower() != 'false'
        self.recording_fps = max(0.5, float(Config.LIVE_RECORDING_FPS))
        self.recording_chunk_seconds = max(30, int(Config.LIVE_RECORDING_CHUNK_SECONDS))
        self.recording_retention_hours = max(1, int(Config.LIVE_RECORDING_RETENTION_HOURS))
        self._recording_interval = 1.0 / self.recording_fps
        self._last_recorded_at = 0.0
        self._recording_writer = None
        self._recording_meta = None
        self._recording_chunk_started_at = None
        self._recording_chunk_prefix = None
        self._recording_frame_size = None
        self._recording_frame_count = 0
        self._last_recording_cleanup = 0.0
        
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()

    def _start_recording_chunk(self, frame_size: Tuple[int, int], now_epoch: float):
        self._close_recording_chunk()
        stamp = datetime.fromtimestamp(now_epoch).strftime('%Y%m%d_%H%M%S')
        source_key = camera_source_key(self.source, self.camera_id)
        prefix = Config.LIVE_RECORDING_DIR / f"live_{source_key}_{stamp}_{int(now_epoch * 1000)}"
        video_path = prefix.with_suffix('.mp4')
        meta_path = prefix.with_suffix('.jsonl')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(video_path), fourcc, self.recording_fps, frame_size)
        if not writer.isOpened():
            logger.error(f"Could not create live recording chunk: {video_path}")
            return
        self._recording_writer = writer
        self._recording_meta = meta_path.open('a', encoding='utf-8')
        self._recording_chunk_started_at = now_epoch
        self._recording_chunk_prefix = prefix
        self._recording_frame_size = frame_size
        self._recording_frame_count = 0
        logger.info(f"Live recording chunk started: {video_path.name}")

    def _close_recording_chunk(self):
        if self._recording_writer is not None:
            try:
                self._recording_writer.release()
            except Exception:
                pass
        if self._recording_meta is not None:
            try:
                self._recording_meta.close()
            except Exception:
                pass
        self._recording_writer = None
        self._recording_meta = None
        self._recording_chunk_started_at = None
        self._recording_chunk_prefix = None
        self._recording_frame_size = None
        self._recording_frame_count = 0

    def _cleanup_old_recordings(self, now_epoch: float):
        if now_epoch - self._last_recording_cleanup < 300:
            return
        self._last_recording_cleanup = now_epoch
        cutoff = now_epoch - (self.recording_retention_hours * 3600)
        for path in Config.LIVE_RECORDING_DIR.glob('live_*.*'):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
            except Exception as e:
                logger.warning(f"Failed to cleanup recording {path}: {e}")

    def _record_frame(self, frame: np.ndarray, detections: List[DetectionResult], tracked_objects: Dict[int, Tuple[float, float]]):
        if not self.recording_enabled:
            return
        now = time.time()
        if now - self._last_recorded_at < self._recording_interval:
            return
        h, w = frame.shape[:2]
        frame_size = (w, h)
        if (
            self._recording_writer is None
            or self._recording_chunk_started_at is None
            or now - self._recording_chunk_started_at >= self.recording_chunk_seconds
            or self._recording_frame_size != frame_size
        ):
            self._start_recording_chunk(frame_size, now)
        if self._recording_writer is None or self._recording_meta is None:
            return

        self._last_recorded_at = now
        self._recording_writer.write(frame)
        entry = {
            'ts': datetime.fromtimestamp(now).isoformat(timespec='milliseconds'),
            'epoch_ms': int(now * 1000),
            'frame_index': self._recording_frame_count,
            'camera_id': self.camera_id,
            'source': str(self.source),
            'source_kind': self.source_kind,
            'width': w,
            'height': h,
            'line_position': self.line_position,
            'session_in': self.session_in,
            'session_out': self.session_out,
            'detections': [
                {
                    'label': det.label,
                    'confidence': round(float(det.confidence), 4),
                    'bbox': [int(v) for v in det.bbox],
                }
                for det in detections
            ],
            'tracks': [
                {'id': int(oid), 'cx': round(float(cx), 2), 'cy': round(float(cy), 2)}
                for oid, (cx, cy) in tracked_objects.items()
            ],
        }
        self._recording_meta.write(json.dumps(entry, separators=(',', ':')) + '\n')
        self._recording_meta.flush()
        self._recording_frame_count += 1
        self._cleanup_old_recordings(now)

    def _update(self):
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                self._frame_counter += 1
                if self._frame_counter % self.frame_skip != 0:
                    with self._frame_lock:
                        if self._processed_frame is None:
                            self._processed_frame = frame.copy()
                    time.sleep(0.005)
                    continue

                h, w = frame.shape[:2]

                # Detect objects using live-adjustable confidence
                _, detections = ml_service.detect(frame, confidence=self.confidence, draw=False)

                processed_frame = frame.copy()

                # Track centroids
                centroids = []
                for det in detections:
                    x1, y1, x2, y2 = det.bbox
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    centroids.append((cx, cy))

                    # Draw bounding box
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(processed_frame, f"{det.label}: {det.confidence:.2f}", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                tracked_objects = self.tracker.update(centroids)
                lp = self.line_position
                line_x = int(w * lp)

                for oid, (cx, cy) in tracked_objects.items():
                    norm_x = cx / w
                    if oid in self.previous_positions:
                        prev_x = self.previous_positions[oid]

                        # Left to right -> OUT (offloaded)
                        if prev_x < lp and norm_x >= lp:
                            if oid not in self.tracker.crossed:
                                self.tracker.crossed.add(oid)
                                self.session_out += 1
                                save_detection_to_db('sugar_bag', 1.0, 'OUT', 0.0)
                                update_inventory_from_detection('Sugar Bag', 'OUT', 1)
                                logger.info(f"🎒 [Live Feed] Bag {oid} crossed OUT")

                        # Right to left -> IN (loaded)
                        elif prev_x > lp and norm_x <= lp:
                            if oid not in self.tracker.crossed:
                                self.tracker.crossed.add(oid)
                                self.session_in += 1
                                save_detection_to_db('sugar_bag', 1.0, 'IN', 0.0)
                                update_inventory_from_detection('Sugar Bag', 'IN', 1)
                                logger.info(f"🎒 [Live Feed] Bag {oid} crossed IN")

                    self.previous_positions[oid] = norm_x

                    # Draw centroid and ID
                    cv2.circle(processed_frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)
                    cv2.putText(processed_frame, f"ID {oid}", (int(cx) - 10, int(cy) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # Draw vertical tracking line
                cv2.line(processed_frame, (line_x, 0), (line_x, h), (255, 0, 0), 2)
                cv2.putText(processed_frame, "TRACKING LINE", (line_x + 10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # Session counter overlay (bottom-left)
                cv2.putText(processed_frame, f"IN:  {self.session_in}", (10, h - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 50), 2)
                cv2.putText(processed_frame, f"OUT: {self.session_out}", (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 80, 255), 2)

                with self._frame_lock:
                    self._processed_frame = processed_frame
                    self._frame_id += 1  # signal encoder cache is stale

                self._record_frame(frame, detections, tracked_objects)
            time.sleep(0.01)

    def get_frame(self):
        with self._frame_lock:
            return self._processed_frame.copy() if self._processed_frame is not None else None

    # ---- low-latency MJPEG helper ----
    # Encodes once per new frame, caches result. Resizes to ≤ 1280px wide for bandwidth.
    _encoded_cache: bytes = None
    _encoded_cache_id: int = 0
    _frame_id: int = 0

    def get_encoded_frame(self) -> bytes | None:
        with self._frame_lock:
            if self._processed_frame is None:
                return None
            # Only re-encode if the frame changed (tracked by _frame_id)
            if self._encoded_cache_id == self._frame_id:
                return self._encoded_cache
            frame = self._processed_frame

        # Resize to max 1280px wide to cut bandwidth (outside lock to not block capture)
        h, w = frame.shape[:2]
        max_w = 1280
        if w > max_w:
            scale = max_w / w
            frame = cv2.resize(frame, (max_w, int(h * scale)), interpolation=cv2.INTER_LINEAR)

        # Encode at quality 70 — good enough for monitoring, ~4-5x smaller than default
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return None
        encoded = buf.tobytes()

        with self._frame_lock:
            self._encoded_cache = encoded
            self._encoded_cache_id = self._frame_id
        return encoded

    def stop(self):
        self._running = False
        try:
            self._thread.join(timeout=1.5)
        except Exception:
            pass
        try:
            self._cap.release()
        except Exception:
            pass
        self._close_recording_chunk()

    def _mark_frame_updated(self):
        """Called inside _update after writing a new processed frame."""
        self._frame_id += 1

camera: Optional[VideoCamera] = None


class CameraManager:
    """Manages one VideoCamera per bay (keyed by bay_id string).
    Falls back to the legacy single `camera` global for the /api/video_feed route.
    """
    def __init__(self):
        self.cameras: Dict[str, VideoCamera] = {}
        self._lock = threading.Lock()

    def get_or_create(self, bay_id: str, source: Any = 0) -> VideoCamera:
        with self._lock:
            if bay_id not in self.cameras or not self.cameras[bay_id]._running:
                logger.info(f"📷 Starting camera for bay {bay_id} (source={source})")
                self.cameras[bay_id] = VideoCamera(source, camera_id=bay_id)
            return self.cameras[bay_id]

    def stop(self, bay_id: str):
        with self._lock:
            cam = self.cameras.pop(bay_id, None)
            if cam:
                cam.stop()
                logger.info(f"📷 Camera stopped for bay {bay_id}")

    def stop_all(self):
        with self._lock:
            for cam in self.cameras.values():
                cam.stop()
            self.cameras.clear()


camera_manager = CameraManager()


class JobStatus(Enum):
    PROCESSING = 'processing'
    COMPLETED = 'completed'
    FAILED = 'failed'

@dataclass
class CompressionJob:
    id: str
    input_path: Path
    output_path: Path
    status: JobStatus = JobStatus.PROCESSING
    original_size: int = 0
    compressed_size: int = 0
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        result = {'job_id': self.id, 'status': self.status.value, 'original_size': self.original_size}
        if self.status == JobStatus.COMPLETED:
            ratio = (1 - self.compressed_size / self.original_size) * 100 if self.original_size > 0 else 0
            result['compressed_size'] = self.compressed_size
            result['ratio'] = round(ratio, 1)
        if self.error: result['error'] = self.error
        return result

compression_jobs: Dict[str, CompressionJob] = {}

# ============================================================================
# FLASK APP
# ============================================================================
