import time


def sleep_to_target(t0, target_fps):
    if target_fps <= 0:
        return
    target_dt = 1.0 / float(target_fps)
    elapsed = time.perf_counter() - t0
    if elapsed < target_dt:
        time.sleep(target_dt - elapsed)
