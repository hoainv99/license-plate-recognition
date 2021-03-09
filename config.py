from dotenv import load_dotenv
from pathlib import Path
import os

#######################################################
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)
#######################################################
MODEL_DETECTOR = "weights/efficient_net_d0.bin"
MODEL_OCR = "weights/multilines_ocr.pth"
TMP = "tmp/"
#######################################################
FORMATE_SUPPORT = [".jpg", ".png", ".jpeg"]
#######################################################
USERNAME = os.getenv("USER_NAME")
PASSWORD = os.getenv("PASSWORD")
