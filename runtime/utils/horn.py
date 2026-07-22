from __future__ import annotations

import threading
import time


class HornBuzzer:
    """horn 이벤트의 rising edge에서 부저를 한 번 울린다.

    state_machine은 horn_on=True/False만 낸다.
    이 클래스는 False->True가 되는 순간에만 별도 thread로 beep pattern을 재생한다.
    """

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.enabled = bool(self.cfg.get("enabled", False))
        self.buzzer = None
        self.was_active = False
        self.lock = threading.Lock()
        self.busy = False

        if not self.enabled:
            print("[horn] disabled")
            return

        try:
            from gpiozero import TonalBuzzer

            self.buzzer = TonalBuzzer(int(self.cfg.get("pin", 12)))
            print(f"[horn] TonalBuzzer enabled pin={self.cfg.get('pin', 12)}")
        except Exception as exc:
            self.enabled = False
            self.buzzer = None
            print(f"[horn] disabled: {exc}")

    def _beep_pattern(self):
        freq = float(self.cfg.get("frequency_hz", 391.0))
        beep_sec = float(self.cfg.get("beep_sec", 0.18))
        gap_sec = float(self.cfg.get("gap_sec", 0.08))
        repeat = max(1, int(self.cfg.get("repeat", 2)))
        try:
            for idx in range(repeat):
                self.buzzer.play(freq)
                time.sleep(beep_sec)
                self.buzzer.stop()
                if idx != repeat - 1:
                    time.sleep(gap_sec)
        finally:
            with self.lock:
                self.busy = False

    def apply(self, active):
        if not self.enabled or self.buzzer is None:
            return
        active = bool(active)
        rising = active and not self.was_active
        self.was_active = active
        if not rising:
            return
        with self.lock:
            if self.busy:
                return
            self.busy = True
        threading.Thread(target=self._beep_pattern, daemon=True).start()

    def close(self):
        if self.buzzer is not None:
            self.buzzer.stop()
