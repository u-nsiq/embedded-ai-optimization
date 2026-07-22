from __future__ import annotations

from dataclasses import dataclass


def clamp(value, low=-1.0, high=1.0):
    return max(low, min(high, float(value)))


@dataclass
class MotorCommand:
    left: float
    right: float
    reason: str = "lane"


def command_from_steer(steer_norm, motor_cfg, reason="lane", speed_scale=1.0, motor_mode="normal", pivot_pwm=None):
    """steer_norm과 speed_scale을 좌우 바퀴 PWM으로 바꾼다.

    steer_norm > 0이면 image-right 방향 조향이다.
    실제 하드웨어 좌/우가 반대면 config.py의 steer_sign만 바꾼다.
    motor_mode="pivot"이면 sign/traffic 강제 회전에서만 제자리 회전을 사용한다.
    """
    steer_norm = clamp(steer_norm, -1.0, 1.0)
    speed_scale = max(0.0, float(speed_scale))

    base_speed = float(motor_cfg.get("base_speed", 0.4)) * speed_scale
    min_speed = float(motor_cfg.get("min_speed", base_speed)) * speed_scale
    max_pwm = float(motor_cfg.get("max_pwm", 0.5))
    steer_to_turn = float(motor_cfg.get("steer_to_turn", 0.3))
    steer_sign = float(motor_cfg.get("steer_sign", 1.0))
    slowdown_at = max(1e-6, float(motor_cfg.get("slowdown_at_abs_steer", 0.7)))
    curve_boost_enabled = bool(motor_cfg.get("curve_boost_enabled", False))
    curve_boost_start = max(0.0, min(0.99, float(motor_cfg.get("curve_boost_start", 0.35))))
    curve_boost_gain = max(0.0, float(motor_cfg.get("curve_boost_gain", 0.0)))
    curve_boost_max = max(0.0, float(motor_cfg.get("curve_boost_max", 0.0)))
    curve_inner_floor_enabled = bool(motor_cfg.get("curve_inner_floor_enabled", False))
    curve_inner_floor_start = max(0.0, min(0.99, float(motor_cfg.get("curve_inner_floor_start", 0.35))))
    curve_inner_min_pwm = max(0.0, float(motor_cfg.get("curve_inner_min_pwm", 0.0)))

    if str(motor_mode) == "pivot" and abs(steer_norm) > 1e-6:
        # 제자리 회전: sign/traffic phase에서만 사용한다.
        # left turn  -> left wheel reverse, right wheel forward
        # right turn -> left wheel forward,  right wheel reverse
        pwm = float(pivot_pwm) if pivot_pwm is not None else base_speed
        pwm = clamp(abs(pwm), 0.0, max_pwm)
        signed_steer = steer_norm * steer_sign
        if signed_steer < 0.0:
            left, right = -pwm, pwm
        else:
            left, right = pwm, -pwm
        return MotorCommand(left, right, f"{reason}:pivot")

    # steer가 커질수록 속도를 min_speed 쪽으로 낮춘다.
    slowdown = min(1.0, abs(steer_norm) / slowdown_at)
    speed = base_speed - (base_speed - min_speed) * slowdown

    turn = steer_norm * steer_sign * steer_to_turn
    left = clamp(speed + turn, -max_pwm, max_pwm)
    right = clamp(speed - turn, -max_pwm, max_pwm)

    # 큰 조향에서는 바깥쪽 바퀴를 추가로 밀어준다.
    # 후처리의 steer_norm은 그대로 두고, 물리 구동만 더 적극적으로 만든다.
    if curve_boost_enabled and abs(steer_norm) > curve_boost_start:
        curve = (abs(steer_norm) - curve_boost_start) / max(1e-6, 1.0 - curve_boost_start)
        boost = min(curve_boost_max, curve_boost_gain * curve)
        if steer_norm * steer_sign < 0.0:
            right = clamp(right + boost, -max_pwm, max_pwm)
        elif steer_norm * steer_sign > 0.0:
            left = clamp(left + boost, -max_pwm, max_pwm)

    # 큰 조향에서 안쪽 바퀴가 너무 죽으면 제자리 회전에 가까워진다.
    # 안쪽 바퀴 최소 PWM을 보장해서 더 큰 원호로 코너를 돌게 만든다.
    if curve_inner_floor_enabled and abs(steer_norm) > curve_inner_floor_start:
        inner_floor = min(max_pwm, curve_inner_min_pwm)
        if steer_norm * steer_sign < 0.0:
            left = clamp(max(left, inner_floor), -max_pwm, max_pwm)
        elif steer_norm * steer_sign > 0.0:
            right = clamp(max(right, inner_floor), -max_pwm, max_pwm)

    return MotorCommand(left, right, reason)


class DifferentialMotor:
    """gpiozero motor output.

    enable_motors=False이면 apply()가 아무것도 하지 않으므로 로그 테스트에 안전하다.
    """

    def __init__(self, motor_cfg):
        self.cfg = dict(motor_cfg)
        self.enabled = bool(self.cfg.get("enable_motors", False))
        self.backend = str(self.cfg.get("backend", "gpiozero"))
        self.forward_mode = str(self.cfg.get("forward_mode", "01"))
        self.PWMA = self.AIN1 = self.AIN2 = None
        self.PWMB = self.BIN1 = self.BIN2 = None

        if not self.enabled:
            print("[motor] disabled")
            return
        if self.backend == "print":
            print("[motor] print backend")
            return
        if self.backend != "gpiozero":
            raise ValueError(f"Unsupported motor backend: {self.backend}")

        from gpiozero import DigitalOutputDevice, PWMOutputDevice

        pins = self.cfg.get("pins", {})
        self.PWMA = PWMOutputDevice(int(pins.get("PWMA", 18)))
        self.AIN1 = DigitalOutputDevice(int(pins.get("AIN1", 22)))
        self.AIN2 = DigitalOutputDevice(int(pins.get("AIN2", 27)))
        self.PWMB = PWMOutputDevice(int(pins.get("PWMB", 23)))
        self.BIN1 = DigitalOutputDevice(int(pins.get("BIN1", 25)))
        self.BIN2 = DigitalOutputDevice(int(pins.get("BIN2", 24)))
        self.stop()
        print("[motor] gpiozero enabled")

    def _set_one(self, pwm, pin1, pin2, speed):
        speed = clamp(speed, -1.0, 1.0)
        if speed == 0:
            pin1.value = 0
            pin2.value = 0
            pwm.value = 0.0
            return

        fwd = (0, 1) if self.forward_mode == "01" else (1, 0)
        if speed > 0:
            v1, v2 = fwd
        else:
            v1, v2 = 1 - fwd[0], 1 - fwd[1]
        pin1.value = v1
        pin2.value = v2
        pwm.value = abs(speed)

    def apply(self, command):
        if not self.enabled:
            return
        if self.backend == "print":
            print(f"[motor] L={command.left:+.3f} R={command.right:+.3f} {command.reason}")
            return
        self._set_one(self.PWMA, self.AIN1, self.AIN2, command.left)
        self._set_one(self.PWMB, self.BIN1, self.BIN2, command.right)

    def stop(self):
        if not self.enabled:
            return
        if self.backend == "print":
            print("[motor] stop")
            return
        self.PWMA.value = 0.0
        self.PWMB.value = 0.0

    def close(self):
        self.stop()
