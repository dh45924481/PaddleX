import paddlex as pdx
from paddlex import create_pipeline
import time

# 创建 OCR pipeline，指定设备为 GPU
pipeline = create_pipeline(pipeline="OCR", device="gpu")

# 记录开始时间
start_time = time.time()

# 使用 OCR pipeline 进行推理
result = pipeline.predict("images/1.png")

# 输出识别结果
for res in result:
    print(res)

# 计算并输出运行时间
end_time = time.time()
run_time = end_time - start_time
print(f"\n推理运行时间: {run_time:.4f} 秒")
