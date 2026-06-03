"""
OpenCV Face Processing Module.

Exposes:
1. FaceAligner - Robust similarity alignment.
2. CLAHEEnhancer - Adaptive contrast optimization for extreme/low-light environments.
3. BlurDetector - High-speed edge variance assessment.
4. FaceQualityEngine - Compiles brightness, focus, tilt, and scale into a quality assessment.
5. BlinkDetector - MediaPipe Eye Aspect Ratio liveness checks.
6. HeadTurnDetector - MediaPipe Face Mesh head turn liveness checks.
"""

from .alignment import FaceAligner
from .enhancement import CLAHEEnhancer
from .blur import BlurDetector
from .fqa import FaceQualityEngine
from .blink import BlinkDetector
from .head_turn import HeadTurnDetector
from .smile import SmileDetector
from .challenge import ChallengeEngine
from .antispoof import PassiveAntiSpoofEngine
from .decision import LivenessDecisionEngine
from .sdk import LivenessSDK

__all__ = [
    "FaceAligner",
    "CLAHEEnhancer",
    "BlurDetector",
    "FaceQualityEngine",
    "BlinkDetector",
    "HeadTurnDetector",
    "SmileDetector",
    "ChallengeEngine",
    "PassiveAntiSpoofEngine",
    "LivenessDecisionEngine",
    "LivenessSDK"
]
