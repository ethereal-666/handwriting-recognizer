import ctypes

# 定义对应C的结构体
class PredictionResult(ctypes.Structure):
    _fields_=[("label",ctypes.c_int),("confidence",ctypes.c_float)]

# 加载文件
lib = ctypes.CDLL('./build/librecognizer.so')

# 指定参数类型
lib.recognizer_create.argtypes=[ctypes.c_char_p]
lib.recognizer_create.restype=ctypes.c_void_p

lib.recognizer_predict.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_float),ctypes.c_int]
lib.recognizer_predict.restype=PredictionResult

lib.recognizer_destroy.argtypes=[ctypes.c_void_p]
lib.recognizer_destroy.restype=None

model_path=b'models/modelv01.bin'

handle=lib.recognizer_create(model_path)

print("handle:",hex(handle))

FloatArray=ctypes.c_float*784
input_data=FloatArray(*([0.0]*784))
result=lib.recognizer_predict(handle,input_data,784)

print("label:",result.label)
print("confidence:",result.confidence)


