# ============================================================
# State machine
# ============================================================
# lane/sign/redline 결과를 최종 행동으로 바꾸는 설정.
# 자주 만질 값은 turn, pending, post-turn, stop 정도다.
STATE = {
    # ----- Event priority -----
    # 같은 loop에서 여러 event가 들어오면 높은 값이 먼저 적용된다.
    "event_priority": {
        "final_redline": 100,  # 도착선은 항상 최우선
        "stop": 90,
        "traffic_red": 85,
        "traffic_green": 75,
        "left": 70,
        "right": 70,
        "straight": 60,
        "speed_20": 50,
        "horn": 40,
    },

    # ----- Race / redline -----
    "redline_arm_sec": 12.0,  # 시작 직후 출발선을 도착선으로 보지 않도록 무시하는 시간

    # ----- Pending before turn -----
    # left/right/traffic event 후 바로 돌지 않고, delay 동안 거의 직진 접근한다.
    # left/right는 따로 튜닝한다. 값이 클수록 표지판이 사라진 뒤 더 앞으로 간 다음 회전한다.
    "left_delay_sec": 3.0,  # left 표지판 사라진 뒤 좌회전 시작 전까지 더 가는 시간
    "right_delay_sec": 4.3,  # right 표지판 사라진 뒤 우회전 시작 전까지 더 가는 시간
    "left_right_delay_sec": 4.0,  # 호환용 fallback. 위 두 값이 없을 때만 사용
    "traffic_light_delay_sec": 1.0,  # green/red event 후 행동 시작 전까지 더 가는 시간
    "pending_max_abs_steer": 0.30,  # pending 중 steer 제한. 흔들리면 낮추고, 길을 못 따라가면 올림
    "pending_speed_scale": 1.0,  # pending 중 고정 속도 배율. 속도 변수 줄이려고 기본 1.0 유지

    # ----- Event turn -----
    # left/right/traffic 모두 curve turn으로 처리한다.
    # curve turn은 일반 조향값을 주고, motor의 curve_boost가 바깥쪽 바퀴를 밀어준다.
    # turn은 시간만으로 끝내지 않고, min 이후 lane이 안정적으로 보이면 종료한다.
    "left_right_turn_mode": "curve",  # left/right 표지판 회전 방식
    "traffic_turn_mode": "curve",  # green/red 신호등 우회전 방식
    "turn_steer": 0.88,  # curve 회전 조향값. 덜 돌면 올리고, 안쪽으로 말리면 낮춤
    "turn_speed_scale": 0.5,  # curve turn 중 전체 속도 배율. 보통 1.0으로 두고 motor/curve_boost에서 속도 튜닝
    "turn_min_sec": 1.2,  # 최소 회전 시간. 이 전에는 lane이 보여도 회전 유지
    "turn_max_sec": 2.7,  # 최대 회전 시간. lane이 안 잡혀도 여기서 종료
    "turn_stable_frames": 2,  # min 이후 lane 안정 frame이 이만큼 쌓이면 curve turn 종료
    "turn_stable_quality": 0.25,  # lane quality가 이 이상이면 안정 후보
    "turn_pivot_pwm": 0.60,  # fallback용. turn mode를 pivot으로 바꿀 때만 사용

    # ----- Post-turn straight -----
    # event turn 후 바로 lane에 맡기지 않고 짧게 직진해서 새 도로축에 맞춘다.
    "post_turn_straight_sec": 0.0,  # 회전 직후 직진 시간. 흔들리면 올리고, 너무 길면 낮춤
    "post_turn_straight_speed_scale": 1.0,  # post-turn 직진 속도. 속도 변수 줄이려고 기본 1.0 유지

    # ----- Stop events -----
    "stop_sec": 3.0,  # stop 표지판 정지 시간
    "traffic_red_stop_sec": 3.0,  # 빨간 신호등 정지 시간. 이후 우회전

    # ----- Straight effect -----
    # straight 표지판이 뜨면 일정 시간 조향 절댓값을 제한한다.
    "straight_hold_sec": 5.0,  # straight 효과 유지 시간
    "straight_max_abs_steer": 0.40,  # 작을수록 더 직진. branch로 새면 낮춤
    "straight_speed_scale": 0.90,  # straight 효과 중 속도 배율

    # ----- Speed limit effect -----
    "speed_20_hold_sec": 3.0,  # speed_20 감속 유지 시간
    "speed_20_scale": 0.650,  # speed_20 감속 배율

    # ----- Horn effect -----
    "horn_sec": 0.7,  # horn event 후 buzzer 유지 시간

    # ----- Safety stop -----
    "emergency_stop_lost_sec": 999.0,  # LANE에서 lane을 너무 오래 잃으면 정지
}
