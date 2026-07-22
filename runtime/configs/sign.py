# ============================================================
# Sign / traffic-light YOLO
# ============================================================
# detector는 bbox 후보를 만들고, trigger는 bbox를 one-shot event로 바꾼다.
#
# event trigger 기준:
#   box_size = max(bbox_width/image_width, bbox_height/image_height)
#   fire_box_size 이상이면 "충분히 가까운 표지판/신호등"으로 본다.
#
# 설정 적용 순서:
#   default -> groups[group] -> classes[class]
#   class에 같은 key를 쓰면 group/default보다 우선한다.
#   roi는 안전하게 deep-merge된다. 예: class roi={"y_max": 0.55}만 써도 나머지 roi 값 유지.
SIGN_MODEL = {
    "enabled": True,
    "selected": "sign_traffic_8class",
    "threads": 1,  # YOLO ORT thread. lane이 CPU를 쓰므로 1 유지
    "input_size": 640,  # YOLO 학습/export 크기
    "conf_threshold": 0.25,  # detector 1차 conf. bbox가 안 뜨면 낮추고, 잡음이 많으면 올림
    "iou_threshold": 0.45,  # NMS IoU. 보통 고정
    "max_det": 3,  # 한 frame 최대 bbox 수
    "classes": ["green", "horn", "left", "red", "right", "speed_20", "stop", "straight"],
    "models": {
        "sign_traffic_8class": {
            "onnx_path": "models/sign/sign_traffic_8class_yolo11n_best.onnx",
        },
    },
}


# ============================================================
# Sign event trigger
# ============================================================
# 튜닝 기준:
#   - 너무 멀리서 event 발생  -> fire_box_size 올림
#   - 바로 앞인데 event 없음  -> fire_box_size 내림, roi 완화, required_hits=1 확인
#   - event 후 행동 이상      -> configs/state.py의 delay/turn/stop/effect 값 수정
#   - 같은 표지판 반복 발화   -> cooldown_sec 올림
#
# 주의:
#   sign.py는 event를 언제 발생시킬지만 정한다.
#   speed_20/straight/horn 지속 시간은 configs/state.py에서만 수정한다.
SIGN_TRIGGER = {
    "enabled": True,

    # ----- Common default -----
    "default": {
        "min_conf": 0.40,  # event 인정 conf. false면 0.50, 놓치면 0.40
        "roi": {
            "x_min": 0.45,  # bbox 중심 cx 하한. 우측 표지판 기준 0.45~0.55
            "x_max": 1.00,
            "y_min": 0.00,
            "y_max": 1.00,
        },
        "required_hits": 1,  # 조건 만족 frame 수. 빠르게 지나가면 1
        "cooldown_sec": 4.0,  # 같은 class 재발화 방지
    },

    # ----- Reusable groups -----
    "groups": {
        "near_sign": {
            "fire_box_size": 0.18,  # left/right/stop 기본 근접 기준
            "required_hits": 1,
            "cooldown_sec": 4.0,
        },
        "near_light": {
            "fire_box_size": 0.50,  # 신호등은 bbox가 크게 잡혀서 높게 둔다
            "required_hits": 1,
            "cooldown_sec": 4.0,
            "roi": {"y_max": 0.55},  # 신호등은 화면 상단부만 허용
        },
        "early_sign": {
            "fire_box_size": 0.14,  # speed_20/straight는 조금 멀리서 받아도 됨
            "required_hits": 1,
            "cooldown_sec": 10.0,
        },
    },

    # ----- Class overrides -----
    # left/right/red/green은 현장에서 가장 자주 만지므로 group과 같은 값도 명시한다.
    "classes": {
        "left": {
            "event_name": "left",
            "group": "near_sign",
            "trigger_mode": "exit_after_arm",  # arm 후 화면/ROI에서 사라지는 순간 event
            "min_conf": 0.45,
            "arm_box_size": 0.14,  # 이 크기 이상 보이면 회전 후보로 기억
            "exit_missing_frames": 1,  # arm 후 YOLO 1회 안 보이면 지나친 것으로 판단
            "required_hits": 1,
            "cooldown_sec": 4.0,
            "roi": {
                "x_min": 0.55,
                "x_max": 1.00,
                "y_min": 0.00,
                "y_max": 0.70,
            },
        },
        "right": {
            "event_name": "right",
            "group": "near_sign",
            "trigger_mode": "exit_after_arm",  # arm 후 화면/ROI에서 사라지는 순간 event
            "min_conf": 0.45,
            "arm_box_size": 0.14,  # 이 크기 이상 보이면 회전 후보로 기억
            "exit_missing_frames": 1,  # arm 후 YOLO 1회 안 보이면 지나친 것으로 판단
            "required_hits": 1,
            "cooldown_sec": 4.0,
            "roi": {
                "x_min": 0.55,
                "x_max": 1.00,
                "y_min": 0.00,
                "y_max": 0.70,
            },
        },
        "red": {
            "event_name": "traffic_red",
            "group": "near_light",
            "min_conf": 0.45,
            "fire_box_size": 0.50,  # 너무 멀리서 멈추면 올림, 지나치면 내림
            "max_top_y_norm": 0.03,  # bbox 윗변이 화면 맨 위에 거의 닿을 때만 trigger
            "required_hits": 1,
            "cooldown_sec": 4.0,
            "roi": {
                "x_min": 0.45,
                "x_max": 1.00,
                "y_min": 0.00,
                "y_max": 0.65,
            },
        },
        "green": {
            "event_name": "traffic_green",
            "group": "near_light",
            "min_conf": 0.45,
            "fire_box_size": 0.50,  # 너무 멀리서 돌면 올림, 지나치면 내림
            "max_top_y_norm": 0.03,  # bbox 윗변이 화면 맨 위에 거의 닿을 때만 trigger
            "required_hits": 1,
            "cooldown_sec": 4.0,
            "roi": {
                "x_min": 0.45,
                "x_max": 1.00,
                "y_min": 0.00,
                "y_max": 0.65,
            },
        },

        "stop": {
            "event_name": "stop",
            "group": "near_sign",
            "cooldown_sec": 10.0,
        },
        "horn": {
            "event_name": "horn",
            "group": "near_sign",
            "fire_box_size": 0.10,
        },
        "speed_20": {
            "event_name": "speed_20",
            "group": "early_sign",
            "cooldown_sec": 10.0,
        },
        "straight": {
            "event_name": "straight",
            "group": "early_sign",
            "cooldown_sec": 8.0,
        },
    },
}
