import urllib.request
import urllib.error
import time
import logging

logger = logging.getLogger(__name__)

def call_api(url, max_retries=3):
	# 设置最大重试次数，避免无限重试
	retry_count = 0
	time.sleep(1)
	url = url.replace(' ', '+')

	while retry_count < max_retries:
		try:
			logger.info(f"Calling API with URL: {url}")
			req = urllib.request.Request(url)
			with urllib.request.urlopen(req) as response:
				call = response.read()
			return call
		except urllib.error.HTTPError as e:
			# 如果遇到HTTP 500错误，进行重试
			if e.code == 500:
				retry_count += 1
				logger.warning(f"HTTP 500 Error encountered. Retry {retry_count}/{max_retries}...")
				time.sleep(5)  # 等待5秒后重试
			else:
				# 对于其他错误，直接记录并返回None
				logger.error(f"Error calling API with URL: {url}")
				logger.error(f"Exception: {str(e)}", exc_info=True)
				return None
		except Exception as e:
			logger.error(f"Error calling API with URL: {url}")
			logger.error(f"Exception: {str(e)}", exc_info=True)
			return None

	# 如果重试次数用完，返回None
	logger.error(f"API call failed after {max_retries} attempts.")
	return None