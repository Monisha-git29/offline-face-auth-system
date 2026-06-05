"""
Security presets configuration for Liveness SDK.
"""

from typing import Dict, Any, List

LOW_SECURITY = {
    "challenges": ["BLINK"],
    "timeout_per_challenge": 8.0,
    "session_timeout": 15.0,
    "passive_history_size": 8,
    "passive_live_threshold": 0.65,
    "passive_spoof_threshold": 0.30,
    "min_face_size_target": 100.0,
    "critical_score_threshold": 10.0,
    "min_confidence": 0.40,
}

MEDIUM_SECURITY = {
    "challenges": ["BLINK", "TURN_LEFT"],
    "timeout_per_challenge": 5.0,
    "session_timeout": 20.0,
    "passive_history_size": 12,
    "passive_live_threshold": 0.75,
    "passive_spoof_threshold": 0.40,
    "min_face_size_target": 150.0,
    "critical_score_threshold": 20.0,
    "min_confidence": 0.50,
}

HIGH_SECURITY = {
    "challenges": ["BLINK", "SMILE", "TURN_LEFT", "TURN_RIGHT"],
    "timeout_per_challenge": 4.0,
    "session_timeout": 25.0,
    "passive_history_size": 15,
    "passive_live_threshold": 0.85,
    "passive_spoof_threshold": 0.50,
    "min_face_size_target": 180.0,
    "critical_score_threshold": 30.0,
    "min_confidence": 0.60,
}

PRESETS = {
    "LOW_SECURITY": LOW_SECURITY,
    "MEDIUM_SECURITY": MEDIUM_SECURITY,
    "HIGH_SECURITY": HIGH_SECURITY,
    # Allow mapping simplified key names
    "LOW": LOW_SECURITY,
    "MEDIUM": MEDIUM_SECURITY,
    "HIGH": HIGH_SECURITY,
}
