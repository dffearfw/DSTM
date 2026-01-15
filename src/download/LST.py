import ee
import os
import time
import requests
import pandas as pd
import logging
from datetime import datetime
import urllib3
from urllib3.exceptions import ProxyError, NewConnectionError
import socket

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ERA5DirectDownloader:
    def __init__(self, use_proxy=False):
        """
        初始化下载器

        Args:
            use_proxy: 是否使用代理
        """
        self.use_proxy = use_proxy
        self.setup_network()
        self.initialize_gee()
        self.china_region = self.get_china_boundary()

    def setup_network(self):
        """设置网络连接，处理代理问题"""
        if self.use_proxy:
            # 设置代理
            proxy_url = "http://127.0.0.1:10809"
            os.environ['HTTP_PROXY'] = proxy_url
            os.environ['HTTPS_PROXY'] = proxy_url
            os.environ['http_proxy'] = proxy_url
            os.environ['https_proxy'] = proxy_url

            logging.info(f"已设置代理: {proxy_url}")

            # 配置requests使用代理
            self.proxies = {
                'http': proxy_url,
                'https': proxy_url
            }

            # 设置socket超时
            socket.setdefaulttimeout(30)
        else:
            # 清除代理设置
            for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
                if key in os.environ:
                    del os.environ[key]
            self.proxies = None
            logging.info("未使用代理")

    def initialize_gee(self):
        """初始化GEE，处理认证问题"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 方法1：尝试直接初始化（使用已有的认证）
                ee.Initialize(project='agile-bonbon-466206-f2')
                logging.info("GEE初始化成功")
                return

            except Exception as e:
                logging.warning(f"GEE初始化尝试 {attempt + 1}/{max_retries} 失败: {e}")

                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    # 所有尝试都失败，尝试重新认证
                    try:
                        logging.info("尝试重新进行GEE认证...")
                        ee.Authenticate()
                        ee.Initialize(project='agile-bonbon-466206-f2')
                        logging.info("GEE重新认证并初始化成功")
                        return
                    except Exception as auth_error:
                        logging.error(f"GEE认证失败: {auth_error}")
                        raise auth_error

    def get_china_boundary(self):
        """获取中国边界"""
        try:
            region = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017") \
                .filter(ee.Filter.eq('country_na', 'China'))
            china_bounds = region.geometry().simplify(maxError=5000).bounds()
            logging.info("成功获取中国边界")
            return china_bounds
        except Exception as e:
            logging.error(f"获取中国边界失败: {e}")
            raise

    def get_download_url_with_retry(self, image, params, max_retries=3):
        """带重试机制的获取下载URL"""
        for attempt in range(max_retries):
            try:
                url = image.getDownloadUrl(params)
                logging.info(f"成功获取下载URL（尝试 {attempt + 1}）")
                return url
            except Exception as e:
                logging.warning(f"获取下载URL失败（尝试 {attempt + 1}/{max_retries}）: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))  # 指数退避
        raise Exception("获取下载URL失败，超过最大重试次数")

    def download_file_with_retry(self, url, output_path, max_retries=5):
        """带重试机制的文件下载"""
        for attempt in range(max_retries):
            try:
                # 配置请求参数
                request_params = {
                    'stream': True,
                    'timeout': 60,
                    'verify': False  # 禁用SSL验证（如果遇到SSL问题）
                }

                if self.proxies:
                    request_params['proxies'] = self.proxies

                logging.info(f"开始下载文件（尝试 {attempt + 1}/{max_retries}）...")

                # 发送请求
                response = requests.get(url, **request_params)
                response.raise_for_status()

                # 获取文件大小
                total_size = int(response.headers.get('content-length', 0))

                # 下载文件
                downloaded = 0
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                if downloaded % (1024 * 1024) == 0:  # 每MB打印一次进度
                                    logging.info(
                                        f"下载进度: {progress:.1f}% ({downloaded / 1024 / 1024:.1f} MB/{total_size / 1024 / 1024:.1f} MB)")

                file_size = os.path.getsize(output_path)
                logging.info(f"文件下载完成: {output_path} ({file_size / 1024 / 1024:.2f} MB)")
                return True

            except requests.exceptions.RequestException as e:
                logging.warning(f"下载失败（尝试 {attempt + 1}/{max_retries}）: {e}")
                if attempt < max_retries - 1:
                    wait_time = 10 * (attempt + 1)  # 指数退避
                    logging.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"下载失败，超过最大重试次数: {e}")
                    return False
            except Exception as e:
                logging.error(f"下载过程中发生错误: {e}")
                return False

    def download_monthly_skin_temperature(self, year, month, output_dir="era5_skin_temperature"):
        """
        下载单个月的ERA5-Land Skin Temperature数据

        Args:
            year: 年份
            month: 月份
            output_dir: 输出目录
        """
        # 创建目录
        os.makedirs(output_dir, exist_ok=True)

        # 构建文件名
        date_str = f"{year}{month:02d}"
        filename = f"ERA5_SkinTemperature_{date_str}.tif"
        output_path = os.path.join(output_dir, filename)

        # 检查文件是否已存在
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            if file_size > 1024:  # 文件大小大于1KB才算有效
                logging.info(f"文件已存在: {filename} ({file_size / 1024 / 1024:.2f} MB)")
                return True, output_path
            else:
                logging.warning(f"文件存在但大小异常，重新下载: {filename}")
                os.remove(output_path)

        logging.info(f"开始下载 {year}-{month:02d} 的数据...")

        try:
            # 构建日期范围
            start_date = f"{year}-{month:02d}-01"
            if month == 12:
                end_date = f"{year + 1}-01-01"
            else:
                end_date = f"{year}-{month + 1:02d}-01"

            # 获取ERA5-Land数据
            logging.info(f"获取数据: {start_date} 到 {end_date}")
            dataset = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
                .filterDate(start_date, end_date) \
                .select('skin_temperature')

            # 计算月平均值
            logging.info("计算月平均值...")
            monthly_mean = dataset.mean().clip(self.china_region)

            # 下载参数
            download_params = {
                'scale': 25000,  # 25km分辨率
                'region': self.china_region,
                'format': 'GEO_TIFF',
                'crs': 'EPSG:4326'
            }

            # 获取下载URL
            logging.info("获取下载URL...")
            url = self.get_download_url_with_retry(monthly_mean, download_params)

            # 下载文件
            logging.info("开始下载文件...")
            success = self.download_file_with_retry(url, output_path)

            if success:
                logging.info(f"成功下载: {filename}")
                return True, output_path
            else:
                return False, f"下载失败: {filename}"

        except Exception as e:
            logging.error(f"处理 {year}-{month:02d} 时发生错误: {e}")
            return False, str(e)

    def download_yearly_data(self, year, output_dir="era5_skin_temperature"):
        """下载单年所有月份数据"""
        year_dir = os.path.join(output_dir, str(year))
        os.makedirs(year_dir, exist_ok=True)

        results = []
        for month in range(1, 13):
            logging.info(f"处理 {year}-{month:02d}")
            success, result = self.download_monthly_skin_temperature(year, month, year_dir)

            results.append({
                'year': year,
                'month': month,
                'success': success,
                'result': result,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

            # 保存进度
            self.save_progress(results, os.path.join(output_dir, f"progress_{year}.csv"))

            # 月份间延迟
            if month < 12:
                logging.info("等待10秒后处理下一个月...")
                time.sleep(10)

        # 统计结果
        success_count = sum(1 for r in results if r['success'])
        logging.info(f"{year}年下载完成: {success_count}/12 个月成功")

        return results

    def download_range_data(self, start_year, end_year, output_dir="era5_skin_temperature"):
        """下载指定年份范围的数据"""
        all_results = []

        for year in range(start_year, end_year + 1):
            logging.info(f"开始处理 {year} 年数据")

            try:
                results = self.download_yearly_data(year, output_dir)
                all_results.extend(results)

                # 年份间较长延迟
                if year < end_year:
                    logging.info("等待30秒后处理下一年...")
                    time.sleep(30)

            except Exception as e:
                logging.error(f"处理 {year} 年数据时发生错误: {e}")
                # 继续处理下一年

        # 保存总进度
        self.save_progress(all_results, os.path.join(output_dir, "all_progress.csv"))

        return all_results

    def save_progress(self, results, progress_file):
        """保存进度到CSV文件"""
        if results:
            df = pd.DataFrame(results)
            df.to_csv(progress_file, index=False)
            logging.info(f"进度已保存到: {progress_file}")


# 测试函数
def test_connection():
    """测试网络连接和GEE认证"""
    print("=" * 50)
    print("测试网络连接和GEE认证")
    print("=" * 50)

    # 测试1：不使用代理
    print("\n1. 测试不使用代理:")
    try:
        downloader_no_proxy = ERA5DirectDownloader(use_proxy=False)
        print("✓ 不使用代理 - 连接成功")
    except Exception as e:
        print(f"✗ 不使用代理 - 连接失败: {e}")

    # 测试2：使用代理
    print("\n2. 测试使用代理:")
    try:
        downloader_with_proxy = ERA5DirectDownloader(use_proxy=True)
        print("✓ 使用代理 - 连接成功")
        return downloader_with_proxy
    except Exception as e:
        print(f"✗ 使用代理 - 连接失败: {e}")
        return None


# 主程序
def main():
    # 测试连接
    downloader = test_connection()

    if downloader is None:
        print("\n尝试两种方法都失败，请检查：")
        print("1. 网络连接是否正常")
        print("2. 代理设置是否正确（127.0.0.1:10809）")
        print("3. 是否已经运行代理软件")
        print("4. 尝试临时关闭防火墙或杀毒软件")
        return

    print("\n" + "=" * 50)
    print("开始下载ERA5-Land Skin Temperature数据")
    print("=" * 50)

    # 创建输出目录
    output_dir = "./era5_skin_temperature_data"
    os.makedirs(output_dir, exist_ok=True)

    # 选项菜单
    print("\n请选择下载选项:")
    print("1. 下载单个月份数据")
    print("2. 下载单年数据")
    print("3. 下载多年数据")
    print("4. 测试下载功能")

    choice = input("请输入选择 (1-4): ").strip()

    if choice == "1":
        # 下载单个月份
        year = int(input("请输入年份 (如: 2020): "))
        month = int(input("请输入月份 (1-12): "))
        downloader.download_monthly_skin_temperature(year, month, output_dir)

    elif choice == "2":
        # 下载单年数据
        year = int(input("请输入年份 (如: 2020): "))
        downloader.download_yearly_data(year, output_dir)

    elif choice == "3":
        # 下载多年数据
        start_year = int(input("请输入开始年份 (如: 2020): "))
        end_year = int(input("请输入结束年份 (如: 2022): "))
        downloader.download_range_data(start_year, end_year, output_dir)

    elif choice == "4":
        # 测试下载功能
        print("\n测试下载功能...")
        try:
            success, result = downloader.download_monthly_skin_temperature(2020, 1, output_dir)
            if success:
                print(f"✓ 测试下载成功: {result}")
            else:
                print(f"✗ 测试下载失败: {result}")
        except Exception as e:
            print(f"✗ 测试过程中发生错误: {e}")

    print("\n" + "=" * 50)
    print("程序执行完成")
    print(f"数据保存在: {os.path.abspath(output_dir)}")
    print("=" * 50)


if __name__ == "__main__":
    # 如果遇到代理问题，可以尝试以下方法：

    # 方法1：直接运行，让程序自动选择
    main()

    # 方法2：手动指定是否使用代理
    # downloader = ERA5DirectDownloader(use_proxy=True)  # 使用代理
    # downloader = ERA5DirectDownloader(use_proxy=False) # 不使用代理

    # 方法3：如果代理有问题，临时修改hosts文件
    # 在C:\Windows\System32\drivers\etc\hosts中添加：
    # 142.250.4.95 oauth2.googleapis.com
    # 142.250.4.95 accounts.google.com