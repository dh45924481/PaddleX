import cv2
import paddleclas
import numpy as np
import io
import time
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn

# 初始化FastAPI应用
app = FastAPI()

# 加载 PaddleClas 模型（全局变量）
image_orientation_predictor = None

# 定义旋转代码映射
cv_rotate_code = {
    "90": cv2.ROTATE_90_COUNTERCLOCKWISE,
    #'180': cv2.ROTATE_180,
    "270": cv2.ROTATE_90_CLOCKWISE,
}


@app.post("/check_flip")
async def check_flip(request: Request):
    """接收原始二进制数据并检查是否需要翻转"""

    # 获取上传的图像二进制数据
    img_bytes = await request.body()

    # 转换为 numpy 数组
    img_np = np.frombuffer(img_bytes, np.uint8)

    # 解码为图像
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if img is None:
        return {"message": "解码图像失败！"}

    # 方向预测
    tic = time.time()
    cls_result = image_orientation_predictor.predict(input_data=img)
    cls_res = next(cls_result)
    angle = cls_res[0]["label_names"][0]
    toc = time.time()

    # 打印预测结果和耗时
    print(f"预测结果：{angle}，耗时：{toc - tic:.4f}秒")

    # 判断是否需要翻转图像
    if angle in cv_rotate_code:
        rotated_img = cv2.rotate(img, cv_rotate_code[angle])  # 进行旋转

        # 编码为PNG图像
        is_success, encoded_img = cv2.imencode(".png", rotated_img)
        if is_success:
            img_byte_arr = io.BytesIO(encoded_img)
            img_byte_arr.seek(0)
            return StreamingResponse(img_byte_arr, media_type="image/png")
        else:
            return {"message": "Failed to encode rotated image."}
    else:
        return {"message": "No flip needed"}


@app.middleware("http")
async def lifespan_middleware(request: Request, call_next):
    async def lifespan():
        global image_orientation_predictor
        try:
            print("开始加载 PaddleClas 模型...")
            image_orientation_predictor = paddleclas.PaddleClas(
                model_name="text_image_orientation",
                use_gpu=False,
                inference_model_dir="./models/text_image_orientation/",
            )
            print("PaddleClas 模型加载完成")
            response = await call_next(request)
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
        finally:
            if image_orientation_predictor:
                del image_orientation_predictor
                print("PaddleClas 模型已释放")
        return response

    return await lifespan()


# 如果是直接运行此脚本，则启动Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8890)
