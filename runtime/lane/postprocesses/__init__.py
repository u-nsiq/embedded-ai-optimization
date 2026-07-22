"""Lane postprocess candidates.

각 후보는 같은 인터페이스를 가진다.

init_memory(cfg) -> dict
update(lanes, memory, cfg) -> LaneSignal dict

LaneSignal 필수 키:
  steer_norm      최종 조향 [-1, 1]
  raw_steer       smoothing 전 조향
  speed_scale     lane이 권장하는 속도 배율
  lane_state      후보 내부 상태 문자열
  stable_forward  상태머신이 lane 복귀 가능성을 판단할 때 쓰는 bool
  quality         lane 신뢰도 0~1
  debug           overlay/log 전용 세부 정보
"""
