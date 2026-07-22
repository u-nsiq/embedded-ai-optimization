# ============================================================
# Runtime
# ============================================================
# target_fps:
#   전체 loop 상한. Pi 전압/열이 불안하면 2.5~3.0부터 시작한다.
#
# sign_every_frames:
#   YOLO를 몇 frame마다 실행할지 정한다.
#   3이면 lane은 매 frame, sign은 3 frame마다 한 번 돈다.
#
# use_lane_speed_scale:
#   True면 lane postprocess가 낸 speed_scale로 위험/불안정 구간에서 자동 감속한다.
#
# redline_every_frames:
#   HSV redline detector를 몇 frame마다 실행할지. 가볍기 때문에 1을 기본으로 둔다.
RUNTIME = {
    "target_fps": 8.0,
    "log_every_frames": 5,
    "use_lane_speed_scale": False,
    "sign_every_frames": 3,
    "redline_every_frames": 1,
    "sign_log_max_rows": 4,
}
