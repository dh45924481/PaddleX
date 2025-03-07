# 导入所需的库：
# argparse：用于解析命令行参数。
# uvicorn：用于运行FastAPI应用的ASGI服务器。
# requests：用于发送HTTP请求。
# fitz：用于处理PDF文件。
# io：用于处理字节流。
# multiprocessing：用于创建多进程。
# FastAPI：用于创建Web API的框架。
# File和UploadFile：用于处理文件上传。
# JSONResponse：用于返回JSON格式的响应。

import argparse
import uvicorn
import requests
import fitz
import io
import multiprocessing
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI()

# OCR API 地址池（假设 OCR 服务器运行在 12345-12348 端口）
# 该列表包含多个OCR服务器的地址，用于负载均衡和容错。
OCR_SERVERS = [
    "http://127.0.0.1:12345/pdf_parse",
    "http://127.0.0.1:12346/pdf_parse",
    "http://127.0.0.1:12347/pdf_parse",
    "http://127.0.0.1:12348/pdf_parse",
    "http://127.0.0.1:12349/pdf_parse",
]


# **1. 切分PDF**
def split_pdf_by_pages(pdf, num_parts=5):
    total_pages = pdf.page_count  # 获取PDF的总页数
    pages_per_part = total_pages // num_parts  # 每部分的页数
    remainder = total_pages % num_parts  # 计算剩余页数

    # 切分范围
    parts = []  # 存储每部分的页码范围
    start_page = 0  # 初始化起始页码
    for i in range(num_parts):
        part_size = pages_per_part + (1 if i < remainder else 0)  # 计算当前部分的页数
        end_page = start_page + part_size  # 计算结束页码
        parts.append((start_page, end_page))  # 将当前部分的页码范围添加到列表
        start_page = end_page  # 更新起始页码
    return parts  # 返回切分后的页码范围


# **2. 提取每部分PDF**
def extract_pdf_part(pdf, start_page, end_page):
    new_pdf = fitz.open()  # 创建一个新的PDF文档
    for page_num in range(start_page, end_page):
        new_pdf.insert_pdf(
            pdf, from_page=page_num, to_page=page_num
        )  # 将指定页码的内容插入新文档

    pdf_bytes = io.BytesIO()  # 创建一个字节流对象
    new_pdf.save(pdf_bytes)  # 将新文档保存到字节流中
    new_pdf.close()  # 关闭新文档
    pdf_bytes.seek(0)  # 将字节流的指针移动到开头
    return pdf_bytes  # 返回字节流对象


# **3. 发送OCR请求**
def send_to_ocr(pdf_part, server_url):
    files = {"file": ("part.pdf", pdf_part, "application/pdf")}
    try:
        response = requests.post(server_url, files=files, timeout=60)  # 发送POST请求
        response.raise_for_status()  # 检查请求是否成功
        # 提取JSON响应中的 content 字段
        return response.json().get("content", "")  # 返回OCR结果
    except Exception as e:
        return f"OCR请求失败: {e}"  # 返回错误信息


# **4. 处理整个PDF**
@app.post("/pdf_parse")
async def receive_pdf(file: UploadFile = File(...)):
    try:
        # 读取文件并加载PDF
        contents = await file.read()  # 异步读取上传的文件内容
        pdf = fitz.open(stream=contents, filetype="pdf")  # 打开PDF文件

        # **切分PDF**
        pdf_parts = split_pdf_by_pages(pdf)  # 切分PDF为多个部分

        # **创建进程池并调用OCR**
        with multiprocessing.Pool(processes=len(pdf_parts)) as pool:
            results = []  # 存储OCR请求的结果
            for idx, (start, end) in enumerate(pdf_parts):
                pdf_part_bytes = extract_pdf_part(pdf, start, end)  # 提取当前部分的PDF
                server_url = OCR_SERVERS[
                    idx % len(OCR_SERVERS)
                ]  # 轮询不同的 OCR 服务器
                # 在多进程中调用 send_to_ocr 函数
                result = pool.apply_async(
                    send_to_ocr, (pdf_part_bytes, server_url)
                )  # 异步调用OCR
                results.append(result)  # 将结果添加到列表

            # 获取所有进程的OCR结果
            ocr_results = [r.get() for r in results]  # 获取每个进程的结果

            # 打印每个OCR返回的 content 字符串长度
            for idx, result in enumerate(ocr_results):
                print(
                    f"OCR服务器 {OCR_SERVERS[idx % len(OCR_SERVERS)]} 返回内容的长度: {len(result)}"
                )

            # 合并OCR结果
            final_text = "\n".join(ocr_results)  # 将所有结果合并为一个字符串

            # 手动关闭进程池并等待所有进程结束
            pool.close()  # 关闭进程池
            pool.join()  # 等待所有进程结束

        return JSONResponse(
            content={"message": "文件上传并解析成功", "content": final_text},
            status_code=200,
        )  # 返回成功响应

    except Exception as e:
        return JSONResponse(
            content={"message": "文件上传解析失败", "error": str(e)}, status_code=400
        )  # 返回错误响应


# **5. 运行服务器**
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Start FastAPI server with a specified port."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=11111,
        help="Port to run the server on (default: 11111)",
    )

    args = parser.parse_args()  # 解析命令行参数

    uvicorn.run(
        app, host="127.0.0.1", port=args.port
    )  # 启动FastAPI服务器，监听指定的端口
