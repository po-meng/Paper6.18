import re
import time
import html
import urllib.parse
import urllib.request
from pathlib import Path


# =========================================================
# 1. 基本设置：Toronto Island A
# =========================================================

SAVE_DIR = Path(r"D:\tools\za\paper\Toronto Island A")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR = 1989
END_YEAR = 2003

PROVINCE = "ON"
CLIMATE_ID = "6158665"
STATION_NAME = "Toronto"

# 你给出的 Toronto Island A 页面参数
HLY_RANGE = "1957-02-06|2006-03-27"
DLY_RANGE = "1957-05-01|2006-03-31"
MLY_RANGE = "1957-01-01|2006-03-01"

HOURLY_PAGE_URL = "https://climate.weather.gc.ca/climate_data/hourly_data_e.html"
BULK_URL = "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"

# 下载间隔，避免访问过快
SLEEP_SECONDS = 1.0

# 每个月失败后重试次数
MAX_RETRY = 3


# =========================================================
# 2. URL 构造
# =========================================================

def build_hourly_page_url(year: int, month: int) -> str:
    """
    构造某年某月的网页 URL，对应你给出的 Toronto Island A 页面。
    """
    params = {
        "hlyRange": HLY_RANGE,
        "dlyRange": DLY_RANGE,
        "mlyRange": MLY_RANGE,
        "climate_id": CLIMATE_ID,
        "Prov": PROVINCE,
        "urlExtension": "_e.html",
        "searchType": "stnName",
        "optLimit": "yearRange",
        "StartYear": 1840,
        "EndYear": 2026,
        "selRowPerPage": 25,
        "Line": 54,
        "searchMethod": "contains",
        "Month": month,
        "Day": 1,
        "txtStationName": STATION_NAME,
        "timeframe": 1,
        "Year": year,
    }

    return HOURLY_PAGE_URL + "?" + urllib.parse.urlencode(params)


def expected_file_name(year: int, month: int) -> str:
    """
    保存成前面清洗脚本能识别的格式：
    en_climate_hourly_ON_6158665_01-1989_P1H.csv
    """
    return f"en_climate_hourly_{PROVINCE}_{CLIMATE_ID}_{month:02d}-{year}_P1H.csv"


# =========================================================
# 3. 网络请求
# =========================================================

def fetch_bytes(url: str, timeout: int = 90) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        }
    )

    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def fetch_text(url: str, timeout: int = 90) -> str:
    raw = fetch_bytes(url, timeout=timeout)

    for enc in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass

    return raw.decode("utf-8", errors="ignore")


# =========================================================
# 4. 从页面源码解析 CSV 下载链接或 stationID
# =========================================================

def parse_csv_download_url(page_html: str, page_url: str):
    """
    优先从页面源码里找 CSV 下载链接。
    """
    decoded = html.unescape(page_html)

    patterns = [
        r'href=["\']([^"\']*bulk_data_e\.html[^"\']*format=csv[^"\']*)["\']',
        r'href=["\']([^"\']*format=csv[^"\']*bulk_data_e\.html[^"\']*)["\']',
        r'([^"\']*bulk_data_e\.html\?[^"\']*format=csv[^"\']*)',
    ]

    for pattern in patterns:
        match = re.search(pattern, decoded, flags=re.IGNORECASE)
        if match:
            raw_url = match.group(1)
            return urllib.parse.urljoin(page_url, raw_url)

    return None


def parse_station_id(page_html: str):
    """
    尝试从页面中解析 stationID。
    Environment Canada 的 bulk_data_e 下载通常使用 stationID。
    """
    decoded = html.unescape(page_html)

    patterns = [
        r"stationID=(\d+)",
        r"StationID=(\d+)",
        r'name=["\']stationID["\']\s+value=["\'](\d+)["\']',
        r'name=["\']StationID["\']\s+value=["\'](\d+)["\']',
        r'id=["\']stationID["\'][^>]*value=["\'](\d+)["\']',
        r'id=["\']StationID["\'][^>]*value=["\'](\d+)["\']',
    ]

    for pattern in patterns:
        match = re.search(pattern, decoded, flags=re.IGNORECASE)
        if match:
            return match.group(1)

    return None


def parse_form_inputs_for_bulk(page_html: str):
    """
    尝试提取页面中 bulk_data_e.html 表单里的 hidden 参数。
    有些页面下载按钮不是直接 href，而是 form 提交。
    """
    decoded = html.unescape(page_html)

    # 找到包含 bulk_data_e.html 的 form
    form_match = re.search(
        r'<form[^>]*action=["\']([^"\']*bulk_data_e\.html[^"\']*)["\'][^>]*>(.*?)</form>',
        decoded,
        flags=re.IGNORECASE | re.DOTALL
    )

    if not form_match:
        return {}

    form_body = form_match.group(2)

    inputs = {}

    # 提取 input name/value
    input_pattern = re.compile(
        r'<input[^>]*name=["\']([^"\']+)["\'][^>]*>',
        flags=re.IGNORECASE
    )

    value_pattern = re.compile(
        r'value=["\']([^"\']*)["\']',
        flags=re.IGNORECASE
    )

    for input_match in input_pattern.finditer(form_body):
        input_tag = input_match.group(0)
        name = input_match.group(1)

        value_match = value_pattern.search(input_tag)
        value = value_match.group(1) if value_match else ""

        inputs[name] = value

    return inputs


# =========================================================
# 5. 备用下载 URL
# =========================================================

def build_bulk_url_with_station_id(station_id: str, year: int, month: int) -> str:
    params = {
        "format": "csv",
        "stationID": station_id,
        "Year": year,
        "Month": month,
        "Day": 1,
        "timeframe": 1,
        "submit": "Download Data",
    }

    return BULK_URL + "?" + urllib.parse.urlencode(params)


def build_bulk_url_with_climate_id(year: int, month: int) -> str:
    """
    有些情况下 climate_id 不一定能直接用于 bulk_data_e，
    所以这个只是备用方案。
    """
    params = {
        "format": "csv",
        "climate_id": CLIMATE_ID,
        "Year": year,
        "Month": month,
        "Day": 1,
        "timeframe": 1,
        "submit": "Download Data",
    }

    return BULK_URL + "?" + urllib.parse.urlencode(params)


def build_hourly_page_csv_url(year: int, month: int) -> str:
    """
    备用方式：在 hourly_data_e 页面参数后加 format=csv。
    """
    params = {
        "hlyRange": HLY_RANGE,
        "dlyRange": DLY_RANGE,
        "mlyRange": MLY_RANGE,
        "climate_id": CLIMATE_ID,
        "Prov": PROVINCE,
        "urlExtension": "_e.html",
        "searchType": "stnName",
        "optLimit": "yearRange",
        "StartYear": 1840,
        "EndYear": 2026,
        "selRowPerPage": 25,
        "Line": 54,
        "searchMethod": "contains",
        "Month": month,
        "Day": 1,
        "txtStationName": STATION_NAME,
        "timeframe": 1,
        "Year": year,
        "format": "csv",
    }

    return HOURLY_PAGE_URL + "?" + urllib.parse.urlencode(params)


def build_bulk_url_from_form_inputs(form_inputs: dict, year: int, month: int) -> str:
    """
    如果页面表单里有 stationID 等隐藏字段，则用表单参数构造下载链接。
    """
    params = dict(form_inputs)

    params.update({
        "format": "csv",
        "Year": year,
        "Month": month,
        "Day": 1,
        "timeframe": 1,
        "submit": "Download Data",
    })

    return BULK_URL + "?" + urllib.parse.urlencode(params)


# =========================================================
# 6. CSV 校验
# =========================================================

def looks_like_valid_hourly_csv(content: bytes) -> bool:
    """
    判断下载内容是否像真正的 hourly CSV，而不是 HTML 错误页。
    """
    text = content[:8000].decode("utf-8", errors="ignore").lower()

    if "<html" in text or "<!doctype" in text:
        return False

    # 加拿大气象 hourly CSV 通常含有这些列名
    keywords = [
        "year",
        "month",
        "day",
        "time",
        "wind spd",
    ]

    return all(k in text for k in keywords)


def save_downloaded_csv(url: str, save_path: Path):
    content = fetch_bytes(url, timeout=90)

    if not looks_like_valid_hourly_csv(content):
        raise RuntimeError("下载内容不像有效 hourly CSV，可能返回的是网页或参数错误。")

    save_path.write_bytes(content)


def existing_file_is_valid(file_path: Path) -> bool:
    if not file_path.exists():
        return False

    if file_path.stat().st_size < 1000:
        return False

    try:
        return looks_like_valid_hourly_csv(file_path.read_bytes())
    except Exception:
        return False


# =========================================================
# 7. 下载单个月份
# =========================================================

def download_one_month(year: int, month: int, known_station_id=None, known_form_inputs=None):
    """
    下载单个月份。
    优先级：
    1. 页面中解析出的 CSV 真实链接；
    2. 页面表单参数；
    3. stationID bulk；
    4. climate_id bulk；
    5. hourly_data_e + format=csv。
    """
    file_name = expected_file_name(year, month)
    save_path = SAVE_DIR / file_name

    if existing_file_is_valid(save_path):
        print(f"[跳过] {year}-{month:02d} 已存在且有效：{file_name}")
        return True, known_station_id, known_form_inputs, "skipped"

    page_url = build_hourly_page_url(year, month)

    station_id = known_station_id
    form_inputs = known_form_inputs if known_form_inputs is not None else {}
    candidate_urls = []

    try:
        page_html = fetch_text(page_url, timeout=90)

        parsed_csv_url = parse_csv_download_url(page_html, page_url)
        if parsed_csv_url:
            candidate_urls.append(("parsed_csv_link", parsed_csv_url))

        parsed_station_id = parse_station_id(page_html)
        if parsed_station_id:
            station_id = parsed_station_id

        parsed_form_inputs = parse_form_inputs_for_bulk(page_html)
        if parsed_form_inputs:
            form_inputs = parsed_form_inputs

    except Exception as e:
        print(f"  网页解析失败，继续使用备用方式：{e}")

    if form_inputs:
        candidate_urls.append(
            ("bulk_form_inputs", build_bulk_url_from_form_inputs(form_inputs, year, month))
        )

    if station_id:
        candidate_urls.append(
            ("stationID_bulk", build_bulk_url_with_station_id(station_id, year, month))
        )

    candidate_urls.append(
        ("climate_id_bulk", build_bulk_url_with_climate_id(year, month))
    )

    candidate_urls.append(
        ("hourly_page_format_csv", build_hourly_page_csv_url(year, month))
    )

    last_error = None

    for method_name, url in candidate_urls:
        for attempt in range(1, MAX_RETRY + 1):
            try:
                print(f"  尝试：{method_name}，第 {attempt} 次")
                save_downloaded_csv(url, save_path)
                print(f"  成功保存：{save_path}")
                return True, station_id, form_inputs, method_name

            except Exception as e:
                last_error = e
                print(f"  失败：{e}")
                time.sleep(0.5)

    print(f"  [失败] {year}-{month:02d}，最后错误：{last_error}")
    return False, station_id, form_inputs, str(last_error)


# =========================================================
# 8. 主程序
# =========================================================

def main():
    print("========== TORONTO ISLAND A 小时数据批量下载 ==========")
    print(f"保存路径：{SAVE_DIR}")
    print(f"年份范围：{START_YEAR}-{END_YEAR}")
    print(f"Climate ID：{CLIMATE_ID}")
    print("目标站点：TORONTO ISLAND A")

    total = 0
    success = 0
    skipped = 0
    failed = []

    station_id = None
    form_inputs = {}

    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            total += 1

            print(f"\n========== 下载 {year}-{month:02d} ==========")

            ok, station_id, form_inputs, method = download_one_month(
                year=year,
                month=month,
                known_station_id=station_id,
                known_form_inputs=form_inputs
            )

            if ok:
                if method == "skipped":
                    skipped += 1
                else:
                    success += 1
            else:
                failed.append({
                    "year": year,
                    "month": month,
                    "error": method
                })

            time.sleep(SLEEP_SECONDS)

    print("\n========== 下载任务完成 ==========")
    print(f"理论月份数：{total}")
    print(f"新下载成功数：{success}")
    print(f"跳过已有数：{skipped}")
    print(f"失败数：{len(failed)}")

    if station_id:
        print(f"解析到并使用的 stationID：{station_id}")
    else:
        print("没有解析到 stationID；如果失败较多，需要手动检查页面源码。")

    if form_inputs:
        form_log = SAVE_DIR / "TorontoIslandA_parsed_form_inputs.txt"
        with open(form_log, "w", encoding="utf-8") as f:
            for k, v in form_inputs.items():
                f.write(f"{k} = {v}\n")
        print(f"已保存解析到的表单参数：{form_log}")

    if failed:
        fail_log = SAVE_DIR / "TorontoIslandA_download_failed_log.txt"

        with open(fail_log, "w", encoding="utf-8") as f:
            for item in failed:
                f.write(
                    f"{item['year']}-{item['month']:02d} | {item['error']}\n"
                )

        print(f"失败记录已保存：{fail_log}")
        print("如果失败很多，把失败日志发我，我再帮你改成固定 stationID 下载。")
    else:
        print("全部 180 个月份下载成功。")


if __name__ == "__main__":
    main()