# ============================================================
# Redline color event
# ============================================================
# final redline은 YOLO가 아니라 HSV + contour로 처리한다.
# 이유:
#   - 빨간 종료선은 화면 하단에 넓게 나타나는 단순 색/형태 이벤트다.
#   - stop 표지판 빨간색과 겹칠 수 있으므로 하단 ROI + 가로 긴 contour 조건으로 분리한다.
#
# 현장에서 주로 만질 값:
#   arm_roi       첫 redline 접근을 감지하는 아래쪽 ROI. 너무 빨리 armed되면 y_min을 올린다.
#   stop_roi      armed 이후 정지 기준 redline을 찾는 중하단 ROI. 못 잡으면 범위를 넓힌다.
#   max_stop_sec  stop_roi를 못 봐도 강제로 멈추는 fallback 시간.
REDLINE_EVENT = {
    "enabled": True,

    # HSV red는 hue가 0 근처와 179 근처로 갈라지므로 두 범위를 합친다.
    "hsv_ranges": [
        {"lower": [0, 100, 80], "upper": [12, 255, 255]},
        {"lower": [168, 100, 80], "upper": [179, 255, 255]},
    ],

    # y=0은 화면 위, y=1은 화면 아래.
    # arm_roi에서 먼저 redline을 본 뒤, stop_roi에서 다시 보면 final_redline event를 낸다.
    "arm_roi": [0.78, 1.00],
    "stop_roi": [0.55, 0.78],

    # mask 노이즈 제거 및 가로 선 연결.
    "open_kernel": [3, 3],
    "close_kernel": [3, 3],
    "horizontal_close_kernel": [17, 3],

    # contour shape filter. 단위가 px/ratio라 크게 바꾸지 말고, 필요할 때만 조금 조절한다.
    "min_area": 350,
    "min_width_ratio": 0.07,
    "min_aspect_wh": 2.0,
    "min_center_overlap": 0.10,
    "min_height_px": 3,
    "central_band": [0.35, 0.65],

    # Event trigger.
    # arm: 아래쪽 ROI에서 접근 감지, stop: 중하단 ROI에서 정지선 확인.
    # final_redline 우선순위는 configs/state.py의 event_priority에서 관리한다.
    "arm_vote_window": 3,
    "stop_vote_window": 3,
    "arm_required_hits": 1,
    "stop_required_hits": 1,
    "cooldown_sec": 4.0,

    # arm 직후 같은 빨간선에 바로 멈추지 않도록 최소 대기한다.
    "min_stop_sec": 0.35,

    # stop_roi 검출 실패 시 기존 6초 delay 방식처럼 최후 정지한다.
    "max_stop_sec": 6.0,

    # 이제 redline 모듈이 정지 타이밍을 정하므로 state_machine 추가 지연은 0으로 둔다.
    "stop_delay_sec": 0.0,
}

