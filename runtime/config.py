from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent

# 기존 실행 스크립트들은 계속 `from config import ...`로 사용한다.
# 실제 튜닝 값은 configs/ 아래 역할별 파일에서 수정한다.
from configs.hardware import CAMERA, HORN, MOTOR
from configs.lane import LANE_DECODE, LANE_MODEL
from configs.lane_postprocess import LANE_POSTPROCESS
from configs.sign import SIGN_MODEL, SIGN_TRIGGER
from configs.redline import REDLINE_EVENT
from configs.state import STATE
from configs.runtime import RUNTIME


__all__ = [
    "ROOT_DIR",
    "CAMERA",
    "MOTOR",
    "HORN",
    "LANE_MODEL",
    "LANE_DECODE",
    "LANE_POSTPROCESS",
    "SIGN_MODEL",
    "SIGN_TRIGGER",
    "REDLINE_EVENT",
    "STATE",
    "RUNTIME",
]
