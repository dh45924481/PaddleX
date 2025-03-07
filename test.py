import requests
import time

# 文件路径
file_path = '105.pdf'

# 目标URL
url = 'http://127.0.0.1:11111/pdf_parse'

# 准备文件
files = {'file': open(file_path, 'rb')}

# 请求头部，删除 Content-Type 头部，requests 会自动处理
headers = {
    'accept': 'application/json',
}

# 记录请求前的时间
start_time = time.time()

# 发送POST请求
response = requests.post(url, headers=headers, files=files)

# 记录请求后的时间
end_time = time.time()

# 计算总时间
total_time = end_time - start_time

# 输出请求结果和总时间
print("Response Status Code:", response.status_code)
print("Response Content:", response.json())  # 这里假设返回的是 JSON 数据
print("Total Request Time:", total_time, "seconds")

# 关闭文件
files['file'].close()
