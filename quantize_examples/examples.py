import numpy as np
import numpy.typing as npt


X = np.array(
    [
        [1, 0, 4, -2, 3],
        [-2, 5, 3, -1, -1],
        [1, -4, 5, 2, 0],
    ],
    dtype=np.float16,
)

W = np.array(
    [[1, 0], [0, 2], [-1, 3], [2, 0], [-1, -2]],
    dtype=np.float16,
)

X_OUTLIER = np.array(
    [
        [2, 45, -1, -17, -1],
        [0, 12, 3, -63, 2],
        [-1, 37, -1, -83, 0],
    ],
    dtype=np.float16,
)

W_OUTLIER = np.array([[-1, 0], [2, 0], [0, -2], [3, -2], [-1, 2]])


# Absmax Quantization
def absmax_quantize(
    X: npt.NDArray[np.float16],
) -> tuple[npt.NDArray[np.int8], np.float16]:
    s_x = np.float16(127 / np.max(np.abs(X)))
    q_X = np.round(s_x * X).astype(np.int8)
    return q_X, s_x


def absmax_dequantize(
    X: npt.NDArray[np.int8], s_x: np.float16
) -> npt.NDArray[np.float16]:
    return X.astype(np.float16) / s_x


# Zero Point Quantization
def zeropoint_quantize(
    X: npt.NDArray[np.float16],
) -> tuple[npt.NDArray[np.int8], np.float16, np.int16]:
    dyna = (np.max(X) - np.min(X)).astype(np.float16)
    dyna = 1 if dyna == 0 else dyna
    nd = np.float16(255 / dyna)
    zp = (-np.round(np.min(X) * nd) - 128).astype(np.int16)
    q_X = np.clip(np.round(X * nd + zp), -127, 127).astype(np.int8)
    return q_X, nd, zp


def zeropoint_dequantize(
    X: npt.NDArray[np.int8], nd: np.float16, zp: np.int16
) -> npt.NDArray[np.float16]:
    return (X.astype(np.float16) - zp.astype(np.float16)) / nd


# Row and Col Wise Absmax Quantization
def row_wise_absmax_quantize(
    X: npt.NDArray[np.float16],
) -> tuple[npt.NDArray[np.int8], np.float16]:
    s_x = np.float16(127 / np.max(np.abs(X), axis=1))
    q_X = np.round(s_x.reshape(-1, 1) * X).astype(np.int8)
    return q_X, s_x


def row_wize_absmax_dequantize(
    X: npt.NDArray[np.int8], s_x: np.float16
) -> npt.NDArray[np.float16]:
    return X.astype(np.float16) / s_x.reshape(-1, 1)


def col_wise_absmax_quantize(
    X: npt.NDArray[np.float16],
) -> tuple[npt.NDArray[np.int8], np.float16]:
    s_x = np.float16(127 / np.max(np.abs(X), axis=0))
    q_X = np.round(s_x.reshape(1, -1) * X).astype(np.int8)
    return q_X, s_x


def col_wise_absmax_dequantize(
    X: npt.NDArray[np.int8], s_x: np.float16
) -> npt.NDArray[np.float16]:
    return X.astype(np.float16) / s_x.reshape(1, -1)


# Matrix Multiplication
def fake_int8_mm(
    A: npt.NDArray[np.int8 | np.int16], B: npt.NDArray[np.int8 | np.int16]
) -> npt.NDArray[np.int32]:
    # use a fake int8 MM operation here, results are 'accumulated' in int32
    return np.dot(A.astype(np.int32), B.astype(np.int32))


def regular_mm(
    X: npt.NDArray[np.float16], W: npt.NDArray[np.float16]
) -> npt.NDArray[np.float16]:
    return np.dot(X, W)


def absmax_quantizated_mm(
    X: npt.NDArray[np.float16], W: npt.NDArray[np.float16]
) -> npt.NDArray[np.float16]:
    q_X, s_x = absmax_quantize(X)
    q_W, s_w = absmax_quantize(W)
    C = fake_int8_mm(q_X, q_W)
    return C / (s_x * s_w)


def zeropoint_quantized_mm(
    X: npt.NDArray[np.float16], W: npt.NDArray[np.float16]
) -> npt.NDArray[np.float16]:
    q_X, nd_x, _ = zeropoint_quantize(X)
    q_W, nd_w, _ = zeropoint_quantize(W)

    C = fake_int8_mm(q_X, q_W)
    return C / (nd_x * nd_w)


def llm_int8_mm(
    X: npt.NDArray[np.float16], W: npt.NDArray[np.float16], alpha=2.0
) -> npt.NDArray[np.float16]:
    outlier_cols = np.abs(X.max(axis=0)) > 10 ** (alpha - 1)
    X_o = X[:, outlier_cols]
    X_no = X[:, ~outlier_cols]
    W_o = W[outlier_cols, :]
    W_no = W[~outlier_cols, :]

    q_X_no, s_x = row_wise_absmax_quantize(X_no)
    q_W_no, s_w = col_wise_absmax_quantize(W_no)
    C_no = fake_int8_mm(q_X_no, q_W_no)
    C_no = np.float16(C_no / (s_x.reshape(-1, 1) * s_w.reshape(1, -1)))

    C_o = np.dot(X_o, W_o).astype(np.float16)
    return C_no + C_o


if __name__ == "__main__":
    print("===Quantization and Dequantization===")
    print("Original matrix")
    print(X)

    print("Absmax quantized matrix")
    q_X, s_x = absmax_quantize(X)
    d_X = absmax_dequantize(q_X, s_x)
    print(d_X)

    print("Zeropoint quantized matrix")
    q_X, nd, zp = zeropoint_quantize(X)
    d_X = zeropoint_dequantize(q_X, nd, zp)
    print(d_X)

    print("Row-wise absmax quantized matrix")
    q_X, s_x = row_wise_absmax_quantize(X)
    d_X = row_wize_absmax_dequantize(q_X, s_x)
    print(d_X)

    print("Col-wise absmax quantized matrix")
    q_X, s_x = col_wise_absmax_quantize(X)
    d_X = col_wise_absmax_dequantize(q_X, s_x)
    print(d_X)

    print("\n===Matrix Multiplication===")

    print("Regular matrix multiplication")
    x_normal = regular_mm(X, W)
    print(x_normal)

    print("Absmax quantized matrix multiplication")
    x_absmax = absmax_quantizated_mm(X, W)
    print(x_absmax)

    print("Zeropoint quantized matrix multiplication")
    x_zeropoint = zeropoint_quantized_mm(X, W)
    print(x_zeropoint)

    print("\n===Matrix Multiplication with Outliers===")

    print("Regular matrix multiplication")
    x_normal = regular_mm(X_OUTLIER, W_OUTLIER)
    print(x_normal)

    print("Absmax quantized matrix multiplication")
    x_absmax = absmax_quantizated_mm(X_OUTLIER, W_OUTLIER)
    print(x_absmax)

    print("LLM.int8 matrix multiplication")
    x_llm_int8 = llm_int8_mm(X_OUTLIER, W_OUTLIER)
    print(x_llm_int8)
