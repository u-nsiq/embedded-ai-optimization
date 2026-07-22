from __future__ import annotations

from dataclasses import dataclass


def clamp(value, low=-1.0, high=1.0):
    return max(low, min(high, float(value)))


@dataclass
class MotorCommand:
    left: float
    right: float
    reason: str = "lane"

    def as_dict(self):
        return {"left": float(self.left), "right": float(self.right), "reason": self.reason}


def command_from_steer(steer_norm, motor_cfg, reason="lane"):
    steer_norm = clamp(steer_norm, -1.0, 1.0)
    base_speed = float(motor_cfg.get("base_speed", 0.18))
    min_speed = float(motor_cfg.get("min_speed", base_speed))
    max_pwm = float(motor_cfg.get("max_pwm", 0.45))
    steer_to_turn = float(motor_cfg.get("steer_to_turn", 0.22))
    steer_sign = float(motor_cfg.get("steer_sign", 1.0))
    slowdown_at = max(1e-6, float(motor_cfg.get("slowdown_at_abs_steer", 0.6)))
    slowdown = min(1.0, abs(steer_norm) / slowdown_at)
    speed = base_speed - (base_speed - min_speed) * slowdown
    turn = steer_norm * steer_sign * steer_to_turn
    left = clamp(speed + turn, -max_pwm, max_pwm)
    right = clamp(speed - turn, -max_pwm, max_pwm)
    return MotorCommand(left, right, reason)


class DifferentialMotor:
    def __init__(self, motor_cfg):
        self.cfg = dict(motor_cfg)
        self.enabled = bool(self.cfg.get("enable_motors", False))
        self.backend = str(self.cfg.get("backend", "gpiozero"))
        self.forward_mode = str(self.cfg.get("forward_mode", "01"))
        self.PWMA = self.AIN1 = self.AIN2 = None
        self.PWMB = self.BIN1 = self.BIN2 = None
        if not self.enabled:
            print("[motor] disabled by config. Commands will be logged only.")
            return
        if self.backend == "print":
            print("[motor] print backend enabled. GPIO output is not used.")
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
        print("[motor] gpiozero backend enabled.")

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
