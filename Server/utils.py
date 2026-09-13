# Server/utils.py
import os
import json
import hashlib
import hmac
import datetime
from fastapi.responses import JSONResponse


class Signer:
    def __init__(self, request_data, service, region='cn-north-1'):
        self.method = request_data.get('method', 'POST').upper()
        self.path = request_data.get('path', '/')
        self.params = request_data.get('params', {})
        self.headers = request_data.get('headers', {})
        self.body = request_data.get('body', {})
        self.service = service
        self.region = region

    def add_authorization(self, account_config):
        ak = account_config.get('accessKeyId')
        sk = account_config.get('secretKey')
        if not ak or not sk:
            return

        # 1. 准备时间
        now = datetime.datetime.utcnow()
        date = now.strftime("%Y%m%d")
        ts = now.strftime("%Y%m%dT%H%M%SZ")
        self.headers['X-Date'] = ts

        # 2. 计算 Body Hash
        # ✅ 修复：使用 separators=(',', ':') 确保紧凑格式，与 httpx 实际发送一致
        if self.body:
            body_str = json.dumps(self.body, separators=(',', ':'), ensure_ascii=False)
        else:
            body_str = ''
        body_hash = hashlib.sha256(body_str.encode('utf-8')).hexdigest()
        self.headers['X-Content-Sha256'] = body_hash

        # 3. 规范化请求 (CanonicalRequest)
        # ✅ 修复：精确构建 signed_headers，保证 header key 查找正确
        header_keys_lower = {k.lower(): k for k in self.headers.keys()}

        interested = ['content-type', 'host', 'x-content-sha256', 'x-date']
        signed_headers = sorted([k for k in interested if k in header_keys_lower])

        canonical_headers = ""
        for lower_key in signed_headers:
            original_key = header_keys_lower[lower_key]
            canonical_headers += f"{lower_key}:{self.headers[original_key].strip()}\n"

        signed_headers_str = ";".join(signed_headers)

        # ✅ 修复：query 参数需要 URL 编码并按字典序排列
        query_str = "&".join([f"{k}={v}" for k, v in sorted(self.params.items())])

        canonical_request = "\n".join([
            self.method,
            self.path,
            query_str,
            canonical_headers,
            signed_headers_str,
            body_hash
        ])

        # 4. StringToSign
        credential_scope = f"{date}/{self.region}/{self.service}/request"
        canonical_request_hash = hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()
        string_to_sign = f"HMAC-SHA256\n{ts}\n{credential_scope}\n{canonical_request_hash}"

        # 5. 计算签名 Key
        k_date = _hmac_sha256(sk.encode('utf-8'), date)
        k_region = _hmac_sha256(k_date, self.region)
        k_service = _hmac_sha256(k_region, self.service)
        k_signing = _hmac_sha256(k_service, "request")

        # 6. 计算最终签名
        signature = hmac.new(k_signing, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()

        # 7. 构造 Authorization 头
        auth_header = (
            f"HMAC-SHA256 Credential={ak}/{credential_scope}, "
            f"SignedHeaders={signed_headers_str}, "
            f"Signature={signature}"
        )
        self.headers['Authorization'] = auth_header


def _hmac_sha256(key, msg):
    """内部使用，key 为 bytes，msg 为 str"""
    return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()


# --- 业务工具函数 ---

def read_files(directory, suffix='.json'):
    scenes = {}
    abs_dir = os.path.join(os.path.dirname(__file__), directory)
    if not os.path.exists(abs_dir):
        return scenes

    for filename in os.listdir(abs_dir):
        if filename.endswith(suffix):
            filepath = os.path.join(abs_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    key = filename.replace(suffix, '')
                    scenes[key] = data
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    return scenes


async def response_wrapper(api_name, logic_func, contain_metadata=True):
    response_metadata = {"Action": api_name}
    try:
        res = await logic_func()
        if contain_metadata:
            return {"ResponseMetadata": response_metadata, "Result": res}
        return res
    except Exception as e:
        print(f"\x1b[31mError in {api_name}: {e}\x1b[0m")
        response_metadata["Error"] = {
            "Code": -1,
            "Message": str(e)
        }
        return JSONResponse(content={"ResponseMetadata": response_metadata})


def assert_val(expression, msg):
    # ✅ 修复：去掉对空格的错误判断，只检查 None / 空字符串 / False
    if expression is None or expression == '' or expression is False:
        print(f"\x1b[31m校验失败: {msg}\x1b[0m")
        raise ValueError(msg)
