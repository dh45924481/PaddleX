# 导入所需的库：
# uvicorn：用于运行FastAPI应用的ASGI服务器。
# FastAPI：用于创建Web API的框架。
# File和UploadFile：用于处理文件上传。
# JSONResponse：用于返回JSON格式的响应。
# create_pipeline：从PaddleX库中创建OCR（光学字符识别）管道。
# fitz：用于处理PDF文件。
# Image：用于图像处理。
# io：用于处理字节流。
# numpy：用于数值计算。
# asyncio：用于异步编程。
# argparse：用于解析命令行参数。

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from paddlex import create_pipeline
import fitz
from PIL import Image
import io
import numpy as np
import asyncio
import argparse

app = FastAPI()

# 初始化一个OCR管道，指定使用GPU进行处理。
pipeline = create_pipeline(pipeline="OCR", device="gpu")


def ocrpage(image):
    try:
        ocr_results = pipeline.predict(image, use_doc_unwarping=False)
        ocr_text = " "
        for res in ocr_results:
            text = res["rec_texts"]
            if text:
                for t in text:
                    ocr_text += str(t)
        return ocr_text
    except:
        return ""


# 定义一个函数get_large_images，用于从PDF页面中提取大图像。
def get_large_images(pdf, page, ratio=0.5):
    threshold = ratio * (page.rect.width * page.rect.height)
    large_images = []
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        img_width = img_info[2]
        img_height = img_info[3]
        if img_width * img_height >= threshold:
            base_image = pdf.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            image_np = np.array(image)
            large_images.append(image_np)
    return large_images


# 定义一个异步函数process_page，用于处理PDF的每一页。
async def process_page(page_num, pdf):
    page = pdf.load_page(page_num)
    image_list = get_large_images(pdf, page)
    text = ""

    if len(image_list) > 0:
        # 如果有大图像，使用OCR处理第一个图像；否则，直接提取页面文本。
        text = await asyncio.to_thread(
            ocrpage, image_list[0]
        )  # 这里我们只使用第一个图像
    else:
        text = page.get_text()

    return text


@app.post("/pdf_parse")
async def receive_pdf(file: UploadFile = File(...)):
    try:
        contents = await file.read()  # 异步读取上传的文件内容
        pdf = fitz.open(stream=contents, filetype="pdf")
        text = ""

        # 创建一个任务列表，处理每一页的内容。
        tasks = [process_page(page_num, pdf) for page_num in range(pdf.page_count)]

        # 使用asyncio.gather并发执行所有任务，等待所有任务完成。
        results = await asyncio.gather(*tasks)

        # 将所有页面的结果合并，并返回成功的JSON响应。
        text = "".join(results)
        return JSONResponse(
            content={"message": "文件上传并解析成功", "content": text}, status_code=200
        )
    except Exception as e:
        return JSONResponse(
            content={"message": "文件上传解析失败", "error": str(e)}, status_code=400
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Start FastAPI server with a specified port."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=12345,
        help="Port to run the server on (default: 12345)",
    )

    args = parser.parse_args()

    # 启动FastAPI服务器，监听指定的端口。
    uvicorn.run(app, host="127.0.0.1", port=args.port)
