import ctypes
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageDraw


# ============================================================
# 路径
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

LIB_PATH = ROOT / "build" / "librecognizer.so"
# MODEL_PATH = [ROOT / "models" / "linear" / "modelv01.bin"]
MODEL_PATH=[ROOT / "models" / "mlp" / "l1v02.bin",ROOT / "models" / "mlp" / "l2v02.bin"]

# ============================================================
# C++ API
# ============================================================

class PredictionResult(ctypes.Structure):
    _fields_ = [
        ("label", ctypes.c_int),
        ("confidence", ctypes.c_float),
    ]


lib = ctypes.CDLL(str(LIB_PATH))

lib.recognizer_create.argtypes = [
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.c_int
]
lib.recognizer_create.restype = ctypes.c_void_p

lib.recognizer_predict.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
]
lib.recognizer_predict.restype = PredictionResult

lib.recognizer_destroy.argtypes = [
    ctypes.c_void_p
]
lib.recognizer_destroy.restype = None

encoded_paths=[str(path).encode() for path in MODEL_PATH]
PathArray=ctypes.c_char_p*len(encoded_paths)
path_array=PathArray(*encoded_paths)
handle=lib.recognizer_create(path_array,len(MODEL_PATH))

# handle = lib.recognizer_create(
#     str(MODEL_PATH).encode()
# )


# ============================================================
# 绘图数据
# ============================================================

CANVAS_SIZE = 280
BRUSH_WIDTH = 20

last_x = None
last_y = None

# 真正给模型处理的灰度图
image = Image.new(
    "L",
    (CANVAS_SIZE, CANVAS_SIZE),
    0
)

image_draw = ImageDraw.Draw(image)


# ============================================================
# 鼠标绘图
# ============================================================

def start_draw(event):
    global last_x, last_y

    last_x = event.x
    last_y = event.y

    radius = BRUSH_WIDTH // 2

    # Tkinter 显示
    canvas.create_oval(
        event.x - radius,
        event.y - radius,
        event.x + radius,
        event.y + radius,
        fill="white",
        outline="white",
    )

    # Pillow 图片
    image_draw.ellipse(
        (
            event.x - radius,
            event.y - radius,
            event.x + radius,
            event.y + radius,
        ),
        fill=255,
    )


def draw(event):
    global last_x, last_y

    if last_x is None or last_y is None:
        return

    # Tkinter 上画
    canvas.create_line(
        last_x,
        last_y,
        event.x,
        event.y,
        fill="white",
        width=BRUSH_WIDTH,
        capstyle=tk.ROUND,
        smooth=True,
    )

    # Pillow 上同步画
    image_draw.line(
        (
            last_x,
            last_y,
            event.x,
            event.y,
        ),
        fill=255,
        width=BRUSH_WIDTH,
    )

    # 在当前点补圆，让笔迹更圆滑
    radius = BRUSH_WIDTH // 2

    image_draw.ellipse(
        (
            event.x - radius,
            event.y - radius,
            event.x + radius,
            event.y + radius,
        ),
        fill=255,
    )

    last_x = event.x
    last_y = event.y


# ============================================================
# 清空
# ============================================================

def clear_canvas():
    global image, image_draw
    global last_x, last_y

    canvas.delete("all")

    image = Image.new(
        "L",
        (CANVAS_SIZE, CANVAS_SIZE),
        0
    )

    image_draw = ImageDraw.Draw(image)

    last_x = None
    last_y = None

    result_label.config(
        text="Draw a digit"
    )


# ============================================================
# 图像预处理
# ============================================================

def preprocess():
    bbox = image.getbbox()

    if bbox is None:
        return None

    # 去除大块黑边
    cropped = image.crop(bbox)

    width, height = cropped.size

    # 保持比例，最长边缩放到 20
    scale = min(
        20 / width,
        20 / height
    )

    new_width = max(
        1,
        int(width * scale)
    )

    new_height = max(
        1,
        int(height * scale)
    )

    resized = cropped.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    # 创建最终 28×28 MNIST 风格图片
    final_image = Image.new(
        "L",
        (28, 28),
        0
    )

    offset_x = (28 - new_width) // 2
    offset_y = (28 - new_height) // 2

    final_image.paste(
        resized,
        (offset_x, offset_y)
    )

    return final_image


# ============================================================
# 预测
# ============================================================

def recognize():
    final_image = preprocess()

    if final_image is None:
        result_label.config(
            text="Please draw a digit"
        )
        return

    # 28 × 28 -> 784
    pixels = list(final_image.getdata())

    # 0~255 -> 0~1
    normalized = [
        value / 255.0
        for value in pixels
    ]

    # Python 数组 -> C float[784]
    FloatArray = ctypes.c_float * 784

    input_data = FloatArray(
        *normalized
    )

    result = lib.recognizer_predict(
        handle,
        input_data,
        784
    )

    if result.label < 0:
        result_label.config(
            text="Prediction failed"
        )
        return

    result_label.config(
        text=(
            f"Prediction: {result.label}\n"
            f"Confidence: {result.confidence * 100:.2f}%"
        )
    )


# ============================================================
# 关闭程序
# ============================================================

def close_app():
    lib.recognizer_destroy(handle)
    root.destroy()


# ============================================================
# GUI
# ============================================================

root = tk.Tk()

root.title("Handwriting Recognizer")

canvas = tk.Canvas(
    root,
    width=CANVAS_SIZE,
    height=CANVAS_SIZE,
    bg="black",
)

canvas.pack(
    padx=10,
    pady=10
)

result_label = tk.Label(
    root,
    text="Draw a digit",
    font=("Arial", 18),
)

result_label.pack(
    pady=5
)

recognize_button = tk.Button(
    root,
    text="Recognize",
    command=recognize,
)

recognize_button.pack(
    pady=5
)

clear_button = tk.Button(
    root,
    text="Clear",
    command=clear_canvas,
)

clear_button.pack(
    pady=5
)


# 鼠标
canvas.bind(
    "<Button-1>",
    start_draw
)

canvas.bind(
    "<B1-Motion>",
    draw
)


# 点右上角关闭窗口
root.protocol(
    "WM_DELETE_WINDOW",
    close_app
)


root.mainloop()