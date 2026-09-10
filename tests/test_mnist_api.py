import ctypes


class PredictionResult(ctypes.Structure):
    _fields_ = [
        ("label", ctypes.c_int),
        ("confidence", ctypes.c_float),
    ]


# =========================
# 加载 C++ 动态库
# =========================

lib = ctypes.CDLL("./build/librecognizer.so")

lib.recognizer_create.argtypes = [ctypes.c_char_p]
lib.recognizer_create.restype = ctypes.c_void_p

lib.recognizer_predict.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
]
lib.recognizer_predict.restype = PredictionResult

lib.recognizer_destroy.argtypes = [ctypes.c_void_p]
lib.recognizer_destroy.restype = None


# =========================
# 创建模型
# =========================

handle = lib.recognizer_create(b"models/modelv01.bin")

FloatArray = ctypes.c_float * 784


# =========================
# 打开 MNIST 测试集
# =========================

image_file = open(
    "data/mnist/t10k-images-idx3-ubyte",
    "rb"
)

label_file = open(
    "data/mnist/t10k-labels-idx1-ubyte",
    "rb"
)


# 跳过 IDX 头
image_file.seek(16)
label_file.seek(8)


# =========================
# 测试
# =========================

correct = 0
test_size = 10000

for i in range(test_size):

    # 一张图正好 28 * 28 = 784 bytes
    raw_pixels = image_file.read(784)

    # 一个 label = 1 byte
    raw_label = label_file.read(1)

    label = raw_label[0]

    # 0~255 -> 0~1
    pixels = [
        value / 255.0
        for value in raw_pixels
    ]

    input_data = FloatArray(*pixels)

    result = lib.recognizer_predict(
        handle,
        input_data,
        784
    )

    if result.label == label:
        correct += 1

    if i < 10:
        print(
            f"sample {i} | "
            f"real={label} | "
            f"predict={result.label} | "
            f"confidence={result.confidence * 100:.2f}%"
        )

    if (i + 1) % 1000 == 0:
        print(f"progress: {i + 1}/{test_size}")


accuracy = correct / test_size

print()
print(f"accuracy: {accuracy * 100:.2f}%")


# =========================
# 清理
# =========================

image_file.close()
label_file.close()

lib.recognizer_destroy(handle)