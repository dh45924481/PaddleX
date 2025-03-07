import fitz
import os
import logging
import warnings
from concurrent.futures import ProcessPoolExecutor
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from paddlex import create_pipeline
from PIL import Image
import io
import numpy as np
from contextlib import asynccontextmanager
import subprocess  # For file conversion
import requests  # For calling local flip service
from PIL import Image, ImageEnhance

warnings.filterwarnings("ignore")
import threading  # 用于线程安全
from typing import List
import asyncio
from multiprocessing import Manager

# 配置日志
import logging
import colorlog
import cv2
import time

# 创建自定义的颜色日志格式
formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(asctime)s - %(levelname)-8s%(reset)s - %(message)s",
    log_colors={
        "DEBUG": "blue",
        "INFO": "green",
        "WARNING": "yellow",  # 设置WARNING颜色为黄色
        "ERROR": "red",
        "CRITICAL": "bold_red",
    },
)

# 设置日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # 可以设置为DEBUG来记录所有级别的日志

# 创建控制台输出的处理器
handler = colorlog.StreamHandler()
handler.setFormatter(formatter)

# 添加处理器到日志记录器
logger.addHandler(handler)


# 图片翻转本地服务地址
LOCAL_FLIP_SERVICE_URL = "http://127.0.0.1:8890/check_flip"

# 设定量
# 初始化全局进程池，最多 50 个进程
MAX_PROCESSES = 5
SINGLE_PROCESSES = 2
MAX_CONCURRENT_FILES = 2  # 最多同时处理 5 个文件

# 在初始化时创建跨进程的共享集合
manager = Manager()
active_processes = manager.list()  # 使用 manager.list() 代替 set()
active_files = manager.Value("i", 0)  # 使用 manager.Value 创建一个共享整数
active_files_lock = asyncio.Lock()  # 异步锁
semaphore = asyncio.Semaphore(MAX_PROCESSES)  # 控制最大并发进程数
process_lock = threading.Lock()  # 用于线程安全的锁


class PDFProcessor:
    """进程级PDF处理封装"""

    def __init__(self):
        self.pipeline = None  # 每个进程独立持有模型

    def init_model(self, pdf_bytes=None):
        """子进程初始化方法（每个进程执行一次）"""
        if self.pipeline is None:
            # logger.info(f"子进程 {os.getpid()} 初始化OCR模型...")
            self.pipeline = create_pipeline("OCR", device="gpu")
            # logger.info(f"子进程 {os.getpid()} 模型加载完成")


processor_pool = None  # 全局进程池


def _init_worker(pdf_bytes):
    """工作进程初始化函数"""
    global _processor
    _processor = PDFProcessor()
    _processor.init_model(pdf_bytes)


def enhance_image(image):
    """图像增强：提高对比度"""
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(1.2)  # 增加对比度


import requests


def check_flip_need(image_bytes):
    """调用本地服务检查是否需要翻转图像"""
    try:
        # 确保 LOCAL_FLIP_SERVICE_URL 已设置
        if LOCAL_FLIP_SERVICE_URL is None or LOCAL_FLIP_SERVICE_URL == "":
            raise ValueError("LOCAL_FLIP_SERVICE_URL is not set")

        # 设置请求头，指示发送的是字节流数据
        headers = {"Content-Type": "application/octet-stream"}

        # 发送POST请求，传递图片字节流数据
        response = requests.post(
            LOCAL_FLIP_SERVICE_URL, data=image_bytes, headers=headers
        )

        # 检查返回的响应内容是否是图像
        if response.headers["Content-Type"].startswith("image"):
            # 如果响应是图片，返回图片字节流
            return response.content
        else:
            # 如果返回内容不是图像，返回None
            return None
    except requests.exceptions.RequestException as e:
        # 捕获网络请求的异常
        logger.info(f"请求失败: {e}")
        return None
    except ValueError as ve:
        # 捕获URL未设置时的异常
        logger.info(f"配置错误: {ve}")
        return None
    except Exception as e:
        # 捕获其他所有异常
        logger.info(f"发生错误: {e}")
        return None


def img_to_bytes(img):
    """将图像转换为字节流"""
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")  # 保存为 JPEG 格式，其他格式也可以
    return img_byte_arr.getvalue()


def remove_stamp(image_array):
    # 直接使用传入的图像数组，而不是使用cv2.imread
    image = image_array.copy()  # 确保不直接修改原始图像

    np.set_printoptions(threshold=np.inf)

    hue_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 扩大红色范围，避免遗漏
    low_range = np.array([0, 100, 100])  # H的范围从0到10，用于红色范围
    high_range = np.array([180, 255, 255])  # H的范围从170到180，用于红色范围

    th = cv2.inRange(hue_image, low_range, high_range)

    index1 = th == 255

    img = np.zeros(image.shape, np.uint8)
    img[:, :] = (255, 255, 255)

    img[index1] = image[index1]

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 尝试减少膨胀次数，避免过度膨胀
    kernel = np.ones((5, 5), np.uint8)
    gray = cv2.dilate(~gray, kernel, iterations=2)  # 调整膨胀的次数

    contours, hierarchy = cv2.findContours(
        gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    tmp3 = image.copy()

    # 轮廓筛选部分，返回面积
    def cnt_area(cnt):
        area = cv2.contourArea(cnt)
        return area

    # 排序轮廓并选择最大的两个
    contours = sorted(contours, key=cnt_area, reverse=True)[:2]

    # 只处理前两个最大的轮廓
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)

        # 提取每个轮廓的区域
        red = image[y : y + h, x - 10 : x + w + 10]

        b, g, r = cv2.split(red)

        # 利用大津法自动选择阈值
        thresh, ret = cv2.threshold(r, 0, 255, cv2.THRESH_OTSU)
        # 对阈值进行调整
        filter_condition = int(thresh * 0.9)
        # 移除红色的印章
        ret, th2 = cv2.threshold(r, filter_condition, 255, cv2.THRESH_BINARY)
        red[:, :, 0] = th2
        red[:, :, 1] = th2
        red[:, :, 2] = th2

        tmp3[y : y + h, x - 10 : x + w + 10] = red

    return tmp3  # 返回处理后的图像


def _process_chunk(args):
    """进程任务处理函数"""
    pdf_bytes, chunk_range = args

    process_id = os.getpid()  # 获取当前进程号
    # print(f"使用进程号 {process_id} 处理分块: {chunk_range}")
    with process_lock:  # 使用线程锁确保原子操作
        active_processes.append(process_id)  # 添加到活跃进程集合中

    start, end = chunk_range
    # logger.info(f"打开PDF文件，处理页码范围: {start} 到 {end}")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    results = []

    for page_index in range(start, end + 1):
        # logger.info(f"开始处理第 {page_index} 页")
        page = doc[page_index]

        if _has_large_image(page):

            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

            img_bytes = pix.tobytes("jpg")

            img = Image.open(io.BytesIO(img_bytes))
            # 去印章
            img_array = np.array(img)
            img_no_stamp = remove_stamp(img_array)  # 处理图像去印章
            img = Image.fromarray(img_no_stamp)  # 转回PIL图像
            # 图片增强
            img = enhance_image(img)
            enhanced_img_bytes = img_to_bytes(img)

            # 调用本地服务判断是否需要翻转图像
            #     logger.info(f"调用翻转服务检查第 {page_index} 页图像是否需要翻转")
            flipped_img_bytes = check_flip_need(enhanced_img_bytes)

            if flipped_img_bytes:
                # logger.info(f"第 {page_index} 页图像已翻转")
                img = Image.open(io.BytesIO(flipped_img_bytes))
            else:
                img = img
                # logger.info(f"第 {page_index} 页图像无需翻转")

            img_array = np.array(img)
            # logger.info(f"开始OCR处理第 {page_index} 页")
            ocr_results = _processor.pipeline.predict(img_array)

            # 处理生成器对象：将生成器转换为列表
            ocr_results_list = list(ocr_results)
            # 处理OCR结果，确保每个OCR结果是字符串
            if isinstance(ocr_results_list, list):
                text_parts = []
                for res in ocr_results_list:
                    # 确保rec_text存在并且是字符串
                    if isinstance(res, dict) and "rec_text" in res:
                        text_parts.append(str(res["rec_text"]))
                    else:
                        logger.warning(f"Unexpected OCR result format: {res}")
                text = " ".join(text_parts)
            else:
                text = "OCR result structure is unexpected"
        else:
            text = page.get_text()

        results.append(text)

    doc.close()

    with process_lock:  # 使用线程锁确保原子操作
        active_processes.remove(process_id)  # 处理完后移除进程

    return (chunk_range, results)


def _has_large_image(page, ratio=0.5):
    """判断大图逻辑"""
    area = page.rect.width * page.rect.height
    return any(img[2] * img[3] > ratio * area for img in page.get_images(full=True))


def _generate_chunks(total_pages):
    """生成分块策略（根据页数动态分块，最多10块）"""
    if total_pages <= SINGLE_PROCESSES:
        return [(i, i) for i in range(total_pages)]  # 每页一个分块
    else:
        base = total_pages // SINGLE_PROCESSES
        remainder = total_pages % SINGLE_PROCESSES
        chunks = []
        current = 0
        for i in range(SINGLE_PROCESSES):
            end = current + base + (1 if i < remainder else 0) - 1
            end = min(end, total_pages - 1)
            chunks.append((current, end))
            current = end + 1
        return chunks


# 将doc转为docx和docx转为pdf的代码保持不变
import tempfile
import subprocess
import os
import logging

logger = logging.getLogger(__name__)


def convert_doc_to_docx(doc_bytes):
    """将doc文件字节流转换为docx文件"""
    try:
        # 获取当前工作目录
        current_dir = os.getcwd()

        # 使用tempfile创建临时文件，存储在当前目录
        with tempfile.NamedTemporaryFile(
            suffix=".doc", delete=False, dir=current_dir
        ) as temp_doc:
            temp_doc.write(doc_bytes)
            temp_doc.close()  # 关闭文件，以便LibreOffice能访问
            # logger.info(f"临时DOC文件路径: {temp_doc.name}")

            doc_path = temp_doc.name  # 获取临时文件路径
            docx_path = doc_path.replace(".doc", ".docx")

            # 执行LibreOffice进行文件转换
            try:
                # logger.info(f"开始执行转换命令: libreoffice --headless --convert-to docx {doc_path}")
                subprocess.run(
                    ["libreoffice", "--headless", "--convert-to", "docx", doc_path],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                logger.info(f"成功转换：{doc_path} -> {docx_path}")
            except subprocess.CalledProcessError as e:
                logger.error(f"LibreOffice 转换失败：{e.stderr.decode()}")
                raise

            # 检查转换后的docx文件是否存在
            if not os.path.exists(docx_path):
                logger.error(f"转换后的DOCX文件不存在: {docx_path}")
                raise FileNotFoundError(f"转换后的DOCX文件不存在: {docx_path}")

            # 读取转换后的文件作为字节流
            with open(docx_path, "rb") as docx_file:
                docx_bytes = docx_file.read()
                # logger.info(f"读取到的docx文件字节流长度: {len(docx_bytes)}")

            # 删除临时文件
            os.remove(doc_path)
            os.remove(docx_path)

            # 返回字节流
            return docx_bytes

    except subprocess.CalledProcessError as e:
        logger.error(f"转换失败：{e}")
        return None
    except Exception as e:
        logger.error(f"发生错误: {str(e)}")
        return None


def convert_docx_to_pdf(docx_bytes):
    """将docx字节流转换为pdf文件"""
    try:
        current_dir = os.getcwd()
        # 使用tempfile创建临时DOCX文件
        with tempfile.NamedTemporaryFile(
            suffix=".docx", delete=False, dir=current_dir
        ) as temp_docx:
            temp_docx.write(docx_bytes)
            temp_docx.close()

            docx_path = temp_docx.name
            pdf_path = docx_path.replace(".docx", ".pdf")

            # 执行LibreOffice进行文件转换
            try:
                # logger.info(f"开始执行转换命令: libreoffice --headless --convert-to pdf {docx_path}")
                subprocess.run(
                    ["libreoffice", "--headless", "--convert-to", "pdf", docx_path],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                logger.info(f"成功转换：{docx_path} -> {pdf_path}")
            except subprocess.CalledProcessError as e:
                logger.error(f"LibreOffice 转换失败：{e.stderr.decode()}")
                raise

            # 读取转换后的PDF文件作为字节流
            with open(pdf_path, "rb") as pdf_file:
                pdf_bytes = pdf_file.read()
                # logger.info(f"读取到的PDF字节流长度: {len(pdf_bytes)}")

            # 删除临时文件
            os.remove(docx_path)
            os.remove(pdf_path)

            # 返回字节流
            return pdf_bytes
    except subprocess.CalledProcessError as e:
        logger.error(f"转换失败：{e}")
        return None
    except Exception as e:
        logger.error(f"发生错误: {str(e)}")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor_pool
    # 初始化进程池（保持原逻辑）
    processor_pool = ProcessPoolExecutor(max_workers=MAX_PROCESSES)
    # 预加载模型到所有进程
    init_tasks = []
    for _ in range(MAX_PROCESSES):
        init_tasks.append(processor_pool.submit(_init_worker, None))
    # 等待所有进程初始化完成
    await asyncio.gather(*[asyncio.wrap_future(t) for t in init_tasks])
    logger.info(f"进程池初始化完成，共 {MAX_PROCESSES} 个工作进程")
    yield
    processor_pool.shutdown()


app = FastAPI(lifespan=lifespan)


# 打印当前进程池状态
def print_pool_status(file_name, total_pages, required_processes):
    """打印进程池当前的状态"""
    used_processes = len(active_processes)
    free_processes = MAX_PROCESSES - used_processes
    logger.info("******************************")
    logger.info(f"当前正在处理的文件数: {active_files.value}")
    logger.info(
        f"开始处理文件: {file_name}，总页数: {total_pages}，需要进程数: {required_processes}"
    )
    logger.info(f"当前使用进程数: {used_processes}/{MAX_PROCESSES}")
    logger.info(f"剩余空闲进程数: {free_processes}")


# 处理文件解析并返回结果
async def process_file(file_name, file_bytes):
    """处理文件并返回解析结果（预留许可方案，并在许可不足时打印警告）"""
    try:

        # 打开PDF文件，获取总页数
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        total_pages = len(doc)
        doc.close()

        # 根据总页数生成任务块（每块对应一个进程），比如最多生成10个任务块
        chunks = _generate_chunks(total_pages)
        required_processes = len(chunks)

        # 如果剩余空闲许可不足，则打印警告信息
        # 注意：这里使用 semaphore._value 读取内部剩余许可数（虽然是内部变量，但用于监控调试是可以接受的）
        free_tokens = semaphore._value
        if free_tokens < required_processes:
            logger.warning(
                "剩余空闲线程数 (%d) 小于解析文件所需线程数 (%d)，任务正在排队处理...",
                free_tokens,
                required_processes,
            )

        # 预先从全局信号量中获取所需的许可（这一步会自动等待直到有足够的许可）
        for _ in range(required_processes):
            await semaphore.acquire()

        # 增加当前正在处理的文件数（多文件并发时统计）
        async with active_files_lock:
            active_files.value += 1

        loop = asyncio.get_event_loop()
        # 提交所有任务，不再需要每个任务前单独“预留许可”
        tasks = []
        for chunk in chunks:
            task = loop.run_in_executor(
                processor_pool, _process_chunk, (file_bytes, chunk)
            )
            tasks.append(task)

        # 可以等待一小段时间后打印当前进程池状态
        await asyncio.sleep(0.5)
        print_pool_status(file_name, total_pages, required_processes)

        # 等待所有分块任务完成
        results = await asyncio.gather(*tasks)

        # 合并所有任务的结果
        combined_results = []
        for chunk_range, chunk_res in results:
            combined_results.extend(chunk_res)

        return {
            "status": "success",
            "total_pages": total_pages,
            "results": combined_results,
        }

    except Exception as e:
        logger.error(f"处理文件 {file_name} 失败: {str(e)}")
        raise HTTPException(500, f"文件处理失败: {str(e)}")
    finally:
        # 当前文件处理结束，减少正在处理的文件数
        async with active_files_lock:
            active_files.value -= 1

        # 将之前预留的许可全部释放，供其他任务使用
        for _ in range(required_processes):
            semaphore.release()

        # 如果所有文件均处理完毕，则打印提示
        if active_files.value == 0:

            logger.info("所有文件处理完成！")


@app.post("/pdf_parse")
async def pdf_parse(file: UploadFile = File(...)):
    """处理上传的PDF文件并返回解析结果"""
    file_name = file.filename
    file_bytes = await file.read()

    # 处理文件并等待结果返回
    result = await process_file(file_name, file_bytes)

    return JSONResponse(result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8891, log_level="info")
