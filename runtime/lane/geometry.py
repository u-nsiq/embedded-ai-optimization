"""Lane model contract constants.

이 값들은 12번 학습, 09 ONNX export parity, 10 Pi runtime parity에서 고정된 값이다.
현장 튜닝 대상이 아니므로 config.py로 빼지 않는다.
"""

RAW_W = 1296
RAW_H = 972
CUT_HEIGHT = 445

IMG_W = 800
IMG_H = 320

NUM_PRIORS = 192
N_OFFSETS = 72
N_STRIPS = N_OFFSETS - 1
OUTPUT_DIM = 78

SAMPLE_Y = list(range(971, 444, -20))
IMAGE_CENTER_X = RAW_W / 2.0
HALF_W = RAW_W / 2.0
