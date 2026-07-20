from pathlib import Path

import numpy as np
import cv2
from sklearn import svm
from skimage.morphology import flood

ASSETS = Path(__file__).resolve().parent.parent / "assets"

def ng_correct_sym(x):
    a = 0.295128763562135
    b = 3.67108346109449E-04
    c = 0.492115123110562
    return x * (1 + x / (a + b * x + c * np.sqrt(x)))


def ngg_estimate_struct(img, mean, theta, alpha):
    a_param = mean * theta
    beta = 1. - alpha

    if img.ndim == 3:
        image = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
    elif img.ndim == 2:
        image = img
    else:
        raise ValueError("Недопустимая размерность изображения")

    rows, cols = image.shape
    temp_image = np.zeros((rows, cols)).astype('float32')
    grad_rows = np.zeros((rows, cols)).astype('float32')
    grad_cols = np.zeros((rows, cols)).astype('float32')

    temp_image[0, :] = image[0, :] / beta
    for row in range(1, rows):
        temp_image[row, :] = alpha * temp_image[row - 1, :] + image[row, :]

    single_temp_col = image[rows - 1, :] / beta
    for row in range(rows - 1, -1, -1):
        temp_image[row, :] = temp_image[row, :] + single_temp_col
        single_temp_col = alpha * single_temp_col + image[row, :]

    grad_rows[:, 0] = temp_image[:, 0] / beta
    for col in range(1, cols):
        grad_rows[:, col] = alpha * grad_rows[:, col - 1] + temp_image[:, col]

    single_temp_row = temp_image[:, cols - 1] / beta
    for col in range(cols - 1, 0, -1):
        grad_rows[:, col] = single_temp_row - grad_rows[:, col - 1]
        single_temp_row = alpha * single_temp_row + temp_image[:, col]
    grad_rows[:, 0] = single_temp_row - temp_image[:, 0]

    temp_image[:, 0] = image[:, 0] / beta
    for col in range(1, cols):
        temp_image[:, col] = alpha * temp_image[:, col - 1] + image[:, col]

    single_temp_row = image[:, cols - 1] / beta
    for col in range(cols - 1, -1, -1):
        temp_image[:, col] += single_temp_row
        single_temp_row = alpha * single_temp_row + image[:, col]

    grad_cols[0, :] = temp_image[0, :] / beta
    for row in range(1, rows):
        grad_cols[row, :] = alpha * grad_cols[row - 1, :] + temp_image[row, :]

    single_temp_col = temp_image[rows - 1, :] / beta
    for row in range(rows - 1, 0, -1):
        grad_cols[row, :] = single_temp_col - grad_cols[row - 1, :]
        single_temp_col = alpha * single_temp_col + temp_image[row, :]
    grad_cols[0, :] = single_temp_col - temp_image[0, :]

    scale = (beta ** 2) / 2
    grad_cols *= scale
    grad_rows *= scale

    lambda_h = (a_param - 1.) / (grad_rows[:, 0:cols - 1] ** 2 + theta)
    lambda_v = (a_param - 1.) / (grad_cols[0:rows - 1, :] ** 2 + theta)

    return lambda_h, ng_correct_sym(lambda_v)


def ng_smoothing(image, delta, lambda_h, lambda_v):
    dtype = np.float32
    last_shape = image.shape[:2]

    x_h = np.zeros(last_shape, dtype=dtype)
    q_h = np.zeros(last_shape, dtype=dtype)
    h_h = np.zeros(last_shape[0], dtype=dtype)
    h_v = np.zeros(last_shape[1], dtype=dtype)
    x_v = np.zeros(last_shape, dtype=dtype)
    q_v = np.zeros(last_shape, dtype=dtype)
    q_r = np.zeros(last_shape, dtype=dtype)

    x_h[:, 0] = image[:, 0]
    q_h[:, 0] = delta[:, 0] + np.finfo(dtype).resolution
    for t2 in range(1, last_shape[1]):
        h_h[:] = q_h[:, t2 - 1] / (1. + q_h[:, t2 - 1] / lambda_h[:, t2 - 1])
        q_h[:, t2] = delta[:, t2] + h_h[:]
        x_h[:, t2] = (delta[:, t2] * image[:, t2] + h_h[:] * x_h[:, t2 - 1]) / q_h[:, t2]

    q_r[:, last_shape[1] - 1] = delta[:, last_shape[1] - 1]
    for t2 in range(last_shape[1] - 2, -1, -1):
        x_h[:, t2] = (q_h[:, t2] * x_h[:, t2] + lambda_h[:, t2] * x_h[:, t2 + 1]) / (q_h[:, t2] + lambda_h[:, t2])
        q_r[:, t2] = q_r[:, t2 + 1] / (1. + q_r[:, t2 + 1] / lambda_h[:, t2]) + delta[:, t2]
        q_h[:, t2] = q_h[:, t2] + q_r[:, t2] - delta[:, t2]

    x_v[0, :] = x_h[0, :]
    q_v[0, :] = q_h[0, :] + np.finfo(dtype).resolution
    for t1 in range(1, last_shape[0]):
        h_v[:] = q_v[t1 - 1, :] / (1. + q_v[t1 - 1, :] / lambda_v[t1 - 1, :])
        q_v[t1, :] = q_h[t1, :] + h_v[:]
        x_v[t1, :] = (q_h[t1, :] * x_h[t1, :] + h_v[:] * x_v[t1 - 1, :]) / q_v[t1, :]

    for t1 in range(last_shape[0] - 2, -1, -1):
        x_v[t1, :] = (q_v[t1, :] * x_v[t1, :] + lambda_v[t1, :] * x_v[t1 + 1, :]) / (q_v[t1, :] + lambda_v[t1, :])

    return x_v


def load_svm():
    train_data = cv2.imread(str(ASSETS / "bkgrnd_shadow_corr.png"))
    if train_data is None:
        raise FileNotFoundError(
            f"Не найден файл обучения: {ASSETS / 'bkgrnd_shadow_corr.png'}")
    train_data = cv2.cvtColor(train_data, cv2.COLOR_RGB2BGR)
    train_data = train_data.reshape((-1, 3))

    oc_svm = svm.OneClassSVM(nu=0.00001, kernel="rbf", gamma=0.00029)
    oc_svm.fit(train_data)
    return oc_svm


def collect_contours(bw_source):
    contours, _ = cv2.findContours(
        (255 - bw_source.astype('uint8')), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    result = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 2000:
            x, y, w, h = cv2.boundingRect(contour)
            if max(h / w, w / h) < 50:
                result.append(contour)
    return result


def segment(image_bgr, oc_svm):
    image = cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR)

    img2bin = image[:, :, 2]
    _, bw_image = cv2.threshold(img2bin, 127, 255, cv2.THRESH_OTSU)

    rough = collect_contours(bw_image)

    mask = np.zeros(image.shape[:2], dtype='uint8')
    cv2.drawContours(mask, rough, -1, (1, 0, 0), 100)
    mask = mask.astype('bool')

    if not mask.any():
        return {"binary": np.zeros(image.shape[:2], np.uint8), "leaves": []}

    test_data = image[mask, :]
    score = oc_svm.decision_function(test_data)
    prob_image = 1. / (1. + pow(2.5, score * 100))

    classification = np.zeros(image.shape[:2])
    classification[mask] = prob_image

    lambda_h = np.zeros_like(image[:, :, 0]).astype('float32')
    lambda_v = np.zeros_like(image[:, :, 0]).astype('float32')
    lambda_h[:, :-1], lambda_v[:-1, :] = ngg_estimate_struct(
        image, mean=7, theta=50, alpha=0.2)

    classification_s = ng_smoothing(
        classification, np.ones_like(image[:, :, 0]), lambda_h, lambda_v)

    threshold = 0.35
    ls_mask = np.zeros_like(classification)
    ls_mask[classification_s <= threshold] = True
    ls_mask = np.logical_or(bw_image.astype('bool'), ls_mask)

    footprint = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    bw_corr = flood(ls_mask * 255, (2, 2), footprint=footprint).astype('uint8') * 255

    final_contours = collect_contours(bw_corr)

    binary = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(binary, final_contours, -1, 255, thickness=-1)

    final_contours = sorted(
        final_contours,
        key=lambda c: (cv2.boundingRect(c)[1] // 100, cv2.boundingRect(c)[0]))

    leaves = []
    PAD = 10
    for contour in final_contours:
        x, y, w, h = cv2.boundingRect(contour)
        x0, y0 = max(x - PAD, 0), max(y - PAD, 0)
        x1, y1 = min(x + w + PAD, image.shape[1]), min(y + h + PAD, image.shape[0])

        leaf_mask = np.zeros(image.shape[:2], np.uint8)
        cv2.drawContours(leaf_mask, [contour], -1, 255, thickness=-1)

        crop = image_bgr[y0:y1, x0:x1].copy()
        crop_mask = leaf_mask[y0:y1, x0:x1]
        # фон вырезанного листа делаем белым
        crop[crop_mask == 0] = (255, 255, 255)

        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        form_factor = 4 * np.pi * area / perimeter ** 2 if perimeter > 0 else 0.0

        leaves.append({
            "crop": crop,
            "mask": crop_mask,
            "bbox": (x0, y0, x1 - x0, y1 - y0),
            "area": float(area),
            "perimeter": float(perimeter),
            "form_factor": float(form_factor),
        })

    return {"binary": binary, "leaves": leaves}