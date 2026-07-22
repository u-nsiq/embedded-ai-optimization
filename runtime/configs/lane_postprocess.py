# ============================================================
# Lane postprocess candidates
# ============================================================
# decoder가 낸 lane point들을 최종 steer/speed_scale로 바꾸는 영역.
#
# selected:
#   "slope_push"              08f 후보. slope + safety push. 현재 기본값.
#   "fixed_base"              08b 후보. both=center+slope, single=slope only.
#   "inside_soft"             12/08 후보. center/heading memory와 lost recovery 포함.
#   "stable_center_tangent"   08g 후보. local center line + tangent 방식.
#
# 튜닝 원칙:
#   - 어떤 후보를 쓸지 먼저 selected로 고른다.
#   - 현장에서는 selected 후보 블록만 본다.
#   - 다른 후보 블록 값은 건드리지 않는다.
LANE_POSTPROCESS = {
    "selected": "fixed_base",  # postprocess 후보 선택
    "candidates": {
        "inside_soft": {
            # ----- A. Geometry sample points -----
            "near_y_ratio": 0.82,  # 차 바로 앞쪽 y. push/안전거리 판단 기준
            "mid_y_ratio": 0.72,  # 중간 y. feature 표시와 방향 확인용
            "far_y_ratio": 0.62,  # 위쪽 y. near와 비교해 lane 기울기 계산

            # ----- B. Lane feature filter -----
            "min_points": 4,  # lane point가 이보다 적으면 조향에 사용하지 않음
            "min_y_span_ratio": 0.06,  # lane이 세로로 너무 짧으면 잡음으로 간주
            "max_extrapolate_ratio": 0.20,  # sample y가 lane 범위 밖으로 너무 멀면 제외

            # ----- C. Steering force -----
            "slope_gain": 0.23,  # lane 기울기를 따라가는 힘. 코너 못 돌면 올림
            "push_gain": 0.75,  # 차선이 중앙 band 안에 들어왔을 때 밀어내는 힘
            "safe_distance_ratio": 0.75,  # 중앙 기준 safety band 폭. 선 밟으면 올림
            "max_push": 1.0,  # push 항 최대값. 너무 튀면 낮춤
            "max_steer_norm": 0.85,  # postprocess가 낼 수 있는 steer 상한

            # ----- D. Lane trust weighting -----
            "trust_conf_weight": 0.35,  # lane confidence 반영 비중
            "trust_span_weight": 0.25,  # 길고 point 많은 lane 우대 비중
            "trust_memory_weight": 0.0,  # 이전 기울기와 비슷한 lane 우대 비중
            "trust_center_weight": 0.10,  # 화면 중앙 근처 lane 우대 비중
            "memory_slope_tolerance": 0.75,  # 이전 기울기와 이 정도 차이까지 유사 방향으로 봄

            # ----- E. Smoothing / memory -----
            "slope_memory_alpha": 1.0,  # 이전 기울기 기억 갱신 속도
            "steer_alpha": 1.0,  # 최종 steer 반응 속도. 느리면 올리고 튀면 낮춤

            # ----- F. Lost lane behavior -----
            "lost_steer_alpha": 0.35,  # lane lost 중 steer 갱신 속도
            "lost_hold_frames": 1,  # lane을 잃어도 이전 steer를 유지할 frame 수
            "lost_decay": 0.50,  # lost가 길어질 때 이전 steer를 얼마나 남길지
            "lost_slope_gain": 0.08,  # lost 중 기억된 기울기를 추가 반영하는 힘

            # ----- G. State/speed signal -----
            "risk_state_threshold": 0.35,  # 이 이상이면 lane_state=slope_push_risk
            "stable_quality": 0.35,  # 상태머신이 lane 안정으로 볼 최소 quality
            "stable_risk": 0.60,  # 상태머신이 lane 안정으로 볼 최대 risk
            "min_speed_scale": 0.35,  # lane이 권장하는 최소 속도 배율
            "risk_slowdown": 0.70,  # risk가 높을수록 감속하는 정도
            "low_quality_slowdown": 0.30,  # quality가 낮을수록 감속하는 정도
        },

        "fixed_base": {
            # ----- A. Geometry: 어느 y 위치에서 lane x/기울기를 읽을지 -----
            # y=0은 화면 위, y=1은 화면 아래. 값을 올릴수록 차 바로 앞쪽만 본다.
            "top_y_ratio": 0.83,  # 기울기 계산의 위쪽 점. 코너를 너무 일찍 읽으면 올림
            "mid_y_ratio": 0.90,  # both lane 중앙 x를 계산하는 기준점
            "bottom_y_ratio": 0.95,  # 기울기 계산의 아래쪽 점. 보통 고정
            "camera_center_offset_ratio": 0.00,  # 조향 기준 화면 중앙 보정. 기준을 오른쪽으로 옮기면 양수

            # ----- B. Filter: 너무 짧거나 이상한 lane/pair 제거 -----
            "min_points": 4,  # lane point 최소 개수. 잡음 lane이 많으면 올림
            "min_y_span_ratio": 0.062,  # lane 세로 길이 최소값. 0.062 ~= 60px/972
            "max_y_distance_ratio": 0.154,  # sample y와 lane 범위 간 허용 거리. 0.154 ~= 150px/972
            "pair_gap_min_ratio": 0.30,  # both lane으로 볼 최소 간격. 0.231 ~= 300px/1296
            "pair_gap_max_ratio": 0.95,  # both lane으로 볼 최대 간격

            # ----- C. Steering: lane geometry를 steer_norm으로 바꾸는 힘 -----
            "both_center_weight": 1.85,  # both lane 중앙에서 벗어난 정도. 직선 중앙 유지가 약하면 올림
            "both_slope_weight": 0.10,  # both lane 평균 기울기 힘. 코너를 너무 일찍/세게 돌면 낮춤
            "single_slope_weight": 0.42,  # single lane 기울기 힘. 한쪽 lane 코너를 못 따라가면 올림
            "max_steer_norm": 0.75,  # lane 후처리가 낼 수 있는 steer 상한

            # ----- D. Memory: 새 steer를 얼마나 빨리 반영할지 -----
            # alpha=1이면 즉시 반영, alpha가 낮을수록 이전 steer를 더 유지한다.
            "both_alpha":0.65,  # both lane에서 반응 속도. 흔들리면 낮추고, 둔하면 올림
            "single_alpha": 0.3,  # single lane에서 반응 속도. single 진입 시 튀면 낮춤
            "no_pair_keep_ratio": 0.15,  # pair가 깨졌을 때 이전 steer를 이 비율만 남겨 branch 추종을 막음
            "no_pair_alpha": 0.85,  # no_pair에서 위 keep_ratio를 얼마나 빨리 적용할지
            "lost_alpha": 0.9,  # lost 중 steer 갱신 속도
            "lost_hold_frames": 16,  # lane lost 후 이전 steer를 유지할 frame 수
            "lost_decay": 0.94,  # lost가 길어질 때 steer 감소율. 1.0이면 계속 유지

            # ----- E. State/speed: 상태머신과 motor에 넘기는 안정도/속도 -----
            "stable_quality": 0.15,  # state_machine이 lane 안정으로 볼 quality 기준
            "both_speed_scale": 1.0,  # both lane 속도 배율
            "single_speed_scale": 0.90,  # single lane 속도 배율
            "no_pair_speed_scale": 1.0,  # pair가 깨진 branch/교차로 의심 상황 속도 배율
            "lost_speed_scale": 0.65,  # lane lost 상황 속도 배율
        },

        "inside_soft": {
            # ----- A. Local geometry sample points -----
            "top_y_ratio": 0.80,  # 위쪽 local y. heading 계산용
            "primary_y_ratio": 0.90,  # center/side 후보를 비교하는 기준 y
            "bottom_y_ratio": 1.00,  # 가장 아래쪽 local y
            "safe_offset_ratio": 0.250,  # single lane을 경계로 볼 때 안쪽 목표 offset. 250px/1296 기준

            # ----- B. Lane feature / pair filter -----
            "min_points": 4,
            "min_y_span_ratio": 0.062,  # 60px/972 기준
            "max_extrapolate_ratio": 0.154,  # 150px/972 기준
            "min_pair_gap_ratio": 0.20,  # 300px/1296 기준
            "max_pair_gap_ratio": 1.10,  # 기존 1300px를 화면 폭 안쪽으로 정리

            # ----- C. Steering gains -----
            "position_gain": 0.70,  # center 위치 오차 반영. 차선 중앙을 못 잡으면 올림
            "heading_gain": 0.60,  # tangent 방향 반영. 코너 반응이 약하면 올림
            "steer_gain": 0.90,  # position+heading 전체 조향 배율
            "max_steer_norm": 0.70,

            # ----- D. Memory / unstable blending -----
            "center_alpha": 0.60,  # center_x smoothing
            "heading_alpha": 0.60,  # heading smoothing
            "steer_alpha": 0.60,  # 최종 steer smoothing
            "turn_bias_alpha": 0.30,  # lost recovery가 참고할 최근 회전 방향 기억
            "jump_center_ratio": 0.231,  # 300px/1296 기준. center가 이 이상 튀면 unstable_blend
            "jump_heading": 0.80,

            # ----- E. Lost recovery -----
            "lost_hold_frames": 5,
            "recovery_gain": 0.10,
            "recovery_ramp": 0.03,

            # ----- F. State/speed signal -----
            "stable_quality": 0.35,
            "both_speed_scale": 1.00,
            "single_speed_scale": 0.80,
            "unstable_speed_scale": 0.75,
            "lost_speed_scale": 0.38,
        },

        "stable_center_tangent": {
            # ----- A. Local geometry sample points -----
            "near_y_ratio": 0.96,  # 가장 아래 local y. 너무 낮으면 모델 point가 부족할 수 있음
            "mid_y_ratio": 0.90,  # center line 기준 y
            "far_y_ratio": 0.86,  # local tangent 계산용 위쪽 y
            "lookahead_y_ratio": 0.88,  # center line을 읽는 목표 y. 낮출수록 가까운 곳을 본다
            "camera_center_offset_ratio": 0.00,  # 화면 중앙 기준 보정. 기준을 오른쪽으로 옮기면 양수, 왼쪽은 음수

            # ----- B. Lane feature filter -----
            "min_points": 4,
            "min_y_span_ratio": 0.055,
            "max_extrapolate_ratio": 0.16,

            # ----- C. Pair stability filter -----
            "min_pair_gap_ratio": 0.30,  # 두 lane 간격 최소. 너무 낮으면 같은 선 두 개를 pair로 볼 수 있음
            "max_pair_gap_ratio": 1.12,  # 두 lane 간격 최대. 너무 높으면 멀리 떨어진 잡음 pair 허용
            "max_gap_change_ratio": 0.18,  # near/mid/far gap 변화가 크면 pair_unstable
            "center_fit_residual_ratio": 0.045,  # local center line fit 오차 허용치
            "max_center_jump_ratio": 0.25,  # 이전 center와 너무 튀면 pair_unstable
            "max_heading_diff": 1.80,  # 두 lane tangent 차이가 너무 크면 pair_unstable

            # ----- D. Steering gains -----
            "center_gain": 0.95,  # both_stable에서 center error 반영 강도
            "slope_gain": 0.45,  # both_stable에서 center tangent 반영 강도
            "unstable_slope_gain": 0.40,  # both_unstable에서 평균 tangent 반영 강도
            "single_slope_gain": 0.40,  # single_slope에서 한 lane tangent 반영 강도
            "max_steer_norm": 0.95,

            # ----- E. Memory / smoothing -----
            "steer_alpha": 0.98,  # both_stable 반응 속도
            "unstable_alpha": 0.50,  # both_unstable 반응 속도
            "unstable_blend": 0.35,  # both_unstable에서 새 slope를 얼마나 섞을지
            "single_alpha": 0.40,  # single_slope 반응 속도
            "single_blend": 0.28,  # single_slope에서 새 slope를 얼마나 섞을지
            "lost_alpha": 1.0,
            "lost_hold_frames": 50,
            "lost_decay": 1.0,

            # ----- F. Speed signal -----
            "both_speed_scale": 0.90,
            "unstable_speed_scale": 0.65,
            "single_speed_scale": 0.75,
            "lost_speed_scale": 0.65,
        },
    },
}
