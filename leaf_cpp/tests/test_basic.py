import numpy as np
import cv2
import leaftools


def _circle(size=200):
    img = np.zeros((size, size), np.uint8)
    cv2.circle(img, (size // 2, size // 2), size // 3, 255, -1)
    return img


def _square(size=200):
    img = np.zeros((size, size), np.uint8)
    c, half = size // 2, size // 6
    cv2.rectangle(img, (c - half, c - half), (c + half, c + half), 255, -1)
    return img


def test_cv_version():
    v = leaftools.cv_version()
    assert isinstance(v, str)
    assert v.count(".") >= 2


def test_descriptor_count_and_range():
    res = leaftools.describe_mask(_circle())
    assert res["count"] == 181
    assert res["angles"].min() == 0
    assert res["angles"].max() == 180


def test_descriptor_at_zero_is_one():
    res = leaftools.describe_mask(_square())
    ang = res["angles"]
    jac = res["jaccard_values"]
    i0 = int(np.argmin(np.abs(ang - 0)))
    assert abs(jac[i0] - 1.0) < 1e-6


def test_square_symmetry():
    res = leaftools.describe_mask(_square())
    ang = res["angles"]
    jac = res["jaccard_values"]

    def at(target):
        return jac[int(np.argmin(np.abs(ang - target)))]

    assert abs(at(0) - 1.0) < 0.01
    assert abs(at(90) - 1.0) < 0.02

    assert abs(at(0) - at(90)) < 0.02

    assert at(45) < at(0) - 0.1

    assert abs(at(45) - at(135)) < 0.02

    assert at(45) <= at(30) + 0.02
    assert at(45) <= at(60) + 0.02


def test_invert_flag():
    white_bg = np.full((200, 200), 255, np.uint8)
    cv2.ellipse(white_bg, (100, 100), (66, 25), 0, 0, 360, 0, -1)

    res_inv = leaftools.describe_leaf(white_bg, invert=True)

    black_bg = np.zeros((200, 200), np.uint8)
    cv2.ellipse(black_bg, (100, 100), (66, 25), 0, 0, 360, 255, -1)
    res_noinv = leaftools.describe_leaf(black_bg, invert=False)

    assert res_inv["center"] == res_noinv["center"]
    assert res_inv["area"] == res_noinv["area"]