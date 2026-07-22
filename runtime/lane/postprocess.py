"""Lane postprocess dispatcher.

후처리 후보를 이 파일 하나에서 선택한다.
새 후보를 추가할 때는 lane/postprocesses/ 아래에 같은 인터페이스의 모듈을 만들고
POSTPROCESS_CANDIDATES에 등록하면 된다.

표준 LaneSignal 출력:
  lane_state      후보 내부 상태 문자열
  steer_norm      최종 조향
  raw_steer       smoothing 전 조향
  speed_scale     lane 기반 권장 속도 배율
  stable_forward  상태머신이 강제 회전 후 lane 복귀 가능성을 볼 때 사용
  quality         lane 판단 신뢰도 0~1
  debug           overlay/log 전용 세부 dict
"""

from __future__ import annotations

from .postprocesses import fixed_base, inside_soft, slope_push, stable_center_tangent
from .postprocesses.common import validate_signal


POSTPROCESS_CANDIDATES = {
    "slope_push": slope_push,
    "stable_center_tangent": stable_center_tangent,
    "inside_soft": inside_soft,
    "fixed_base": fixed_base,
}


def selected_postprocess_name(cfg):
    return str(cfg.get("selected", "slope_push"))


def selected_postprocess_cfg(cfg):
    name = selected_postprocess_name(cfg)
    candidates = cfg.get("candidates", {})
    if name not in candidates:
        raise KeyError(f"Unknown lane postprocess '{name}'. Available: {sorted(candidates)}")
    return candidates[name]


def selected_postprocess_module(cfg):
    name = selected_postprocess_name(cfg)
    if name not in POSTPROCESS_CANDIDATES:
        raise KeyError(f"Postprocess module not registered: {name}")
    return POSTPROCESS_CANDIDATES[name]


def init_memory(cfg):
    module = selected_postprocess_module(cfg)
    return module.init_memory(selected_postprocess_cfg(cfg))


def update_lane_postprocess(lanes, memory, cfg):
    name = selected_postprocess_name(cfg)
    module = selected_postprocess_module(cfg)
    params = selected_postprocess_cfg(cfg)
    signal = module.update(lanes, memory, params)
    validate_signal(signal, name)

    signal["postprocess"] = name
    return signal
