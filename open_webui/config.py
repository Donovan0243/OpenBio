import os

####################################
# Load .env file
####################################

try:
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv("./.env"))
except ImportError:
    print("dotenv not installed, skipping...")

# 获取当前文件所在目录的绝对路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

API_KEY = os.getenv("PIPELINES_API_KEY", "0p3n-w3bu!")
# 设置 PIPELINES_DIR 为相对于当前目录的 pipelines 目录
PIPELINES_DIR = os.getenv("PIPELINES_DIR", os.path.join(CURRENT_DIR, "pipelines"))
