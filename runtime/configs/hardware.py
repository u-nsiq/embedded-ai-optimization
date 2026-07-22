# ============================================================
# 0. Camera
# ============================================================
# 평소에는 거의 건드리지 않는 영역이다.
#
# width/height:
#   모델 학습과 ONNX 검증에서 고정한 원본 frame 크기다.
#   lane model은 내부에서 crop 445 -> resize 800x320을 수행한다.
#   카메라 해상도를 낮추는 실험은 가능하지만, 지금 v2 기본값은 검증된 1296x972다.
#
# color_mode:
#   Pi camera가 RGB로 들어오면 "rgb_to_bgr"로 바꾼다.
#   현재 주행 실험에서는 BGR로 맞춰져 있어서 "bgr"을 기본으로 둔다.
#
# rotate_180:
#   카메라가 거꾸로 장착되어 있으면 True.
CAMERA = {
    "backend": "picamera2",
    "width": 1296,
    "height": 972,
    "force_resize_to_raw": True,
    "color_mode": "bgr",
    "rotate_180": True,
    "flip_horizontal": False,
    "flip_vertical": False,
    "startup_sleep_sec": 0.5,
}


# ============================================================
# 1. Motor
# ============================================================
# 주행 현장에서 가장 자주 만지는 값은 base_speed, min_speed,
# steer_to_turn, slowdown_at_abs_steer 네 개다.
#
# enable_motors:
#   실제 바퀴를 굴릴 때 True. 책상 위 로그 테스트는 False.
#
# base_speed:
#   직선 기본 속도. 기준 0.35~0.45.
#   제한시간이 부족하면 올리고, 코너 이탈/전압 경고가 있으면 낮춘다.
#
# min_speed:
#   큰 조향 중에도 유지할 속도. 기준 0.18~0.30.
#   코너에서 거의 멈추면 올리고, 안쪽 선을 밟으면 낮춘다.
#
# steer_to_turn:
#   steer_norm을 좌우 바퀴 속도 차이로 바꾸는 강도. 기준 0.25~0.40.
#   코너를 못 돌면 올리고, 너무 안쪽으로 급하게 돌면 낮춘다.
#
# curve_boost:
#   큰 조향에서 바깥쪽 바퀴를 추가 가속한다.
#   코너가 답답하거나 lost 중 레인을 다시 못 찾으면 gain/max를 올린다.
#   너무 안쪽으로 말리면 gain/max를 낮추거나 start를 올린다.
#
# curve_inner_floor:
#   큰 조향에서 안쪽 바퀴가 너무 느려지지 않도록 최소 PWM을 보장한다.
#   코너가 제자리 회전처럼 말리면 min_pwm을 올린다.
#   회전이 부족해서 바깥으로 밀리면 min_pwm을 낮춘다.
#
# slowdown_at_abs_steer:
#   abs(steer_norm)이 이 값에 가까워질수록 base_speed에서 min_speed로 감속한다.
#   기준 0.55~0.85. 낮출수록 코너에서 빨리 감속한다.
#
# max_pwm:
#   바퀴에 줄 수 있는 PWM 상한. 전압 경고가 반복되면 0.45~0.50 안에서 둔다.
MOTOR = {
    "enable_motors": True,
    "backend": "gpiozero",
    "base_speed": 0.45,
    "min_speed": 0.25,
    "steer_to_turn": 0.38,
    
    "curve_boost_enabled": True,
    "curve_boost_start": 0.32,
    "curve_boost_gain": 0.20,   
    "curve_boost_max": 0.60,
    
    "curve_inner_floor_enabled": True,
    "curve_inner_floor_start": 0.20,
    "curve_inner_min_pwm": 0.25,
    
    "steer_sign": 1.0,
    "slowdown_at_abs_steer": 0.65,
    "max_pwm": 1.0,
    "forward_mode": "01",
    "pins": {
        "PWMA": 18,
        "AIN1": 22,
        "AIN2": 27,
        "PWMB": 23,
        "BIN1": 25,
        "BIN2": 24,
    },
}


# ============================================================
# 1-1. Horn buzzer
# ============================================================
# DaduiNo 예제 기준 부저는 GPIO12 TonalBuzzer를 사용한다.
# horn 표지판 이벤트가 발생하면 아래 패턴을 1회 재생한다.
#
# enabled:
#   Pi에서 실제 부저를 울릴 때 True. gpiozero/TonalBuzzer가 없는 환경이면 False.
#
# beep_sec / gap_sec / repeat:
#   삑 소리 길이, 간격, 반복 횟수. 너무 길면 주행 판단을 방해하므로 짧게 둔다.
HORN = {
    "enabled": True,
    "pin": 12,
    "frequency_hz": 391.0,
    "beep_sec": 0.18,
    "gap_sec": 0.08,
    "repeat": 2,
}
