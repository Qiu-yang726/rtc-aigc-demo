# -*- coding: utf-8 -*-
"""
火山引擎（Volcengine / 火山 RTC）实时音视频 Token 生成器。

对齐官方 Java 实现：
    volcengine.rtc.token.AccessToken / ByteBuf / Utils
参考：https://www.volcengine.com/docs/6348/291043

与原始代码的差异（修复点）：
    官方使用 TreeMap<Short, Integer>（按 key 升序）序列化 privileges，
    本实现把 privileges 按 key 升序打包，避免因插入顺序导致签名不一致。
"""
import time
import struct
import hmac
import hashlib
import base64
import random
from io import BytesIO

VERSION = "001"
VERSION_LENGTH = 3
APP_ID_LENGTH = 24

# 权限定义（与官方 Privileges 枚举一致）
PRIVILEGES = {
    "PrivPublishStream": 0,
    "privPublishAudioStream": 1,
    "privPublishVideoStream": 2,
    "privPublishDataStream": 3,
    "PrivSubscribeStream": 4,
}


class ByteBuf:
    def __init__(self, data=None):
        self.buffer = BytesIO(data) if data else BytesIO()

    def pack(self):
        return self.buffer.getvalue()

    def put_uint16(self, v):
        self.buffer.write(struct.pack('<H', v))
        return self

    def put_uint32(self, v):
        self.buffer.write(struct.pack('<I', v))
        return self

    def put_bytes(self, b):
        self.put_uint16(len(b))
        self.buffer.write(b)
        return self

    def put_string(self, s):
        return self.put_bytes(s.encode('utf-8'))

    def put_tree_map_uint32(self, m):
        if not m:
            self.put_uint16(0)
            return self

        # 关键修复：按 key 升序打包，与官方 TreeMap 保持一致
        self.put_uint16(len(m))
        for k, v in sorted(m.items()):
            self.put_uint16(int(k))
            self.put_uint32(int(v))
        return self


class AccessToken:
    def __init__(self, app_id, app_key, room_id, user_id):
        self.app_id = app_id
        self.app_key = app_key
        self.room_id = room_id
        self.user_id = user_id
        self.issued_at = int(time.time())
        self.nonce = random.randint(0, 0xFFFFFFFF)
        self.expire_at = 0
        self.privileges = {}

    def add_privilege(self, privilege, expire_timestamp):
        self.privileges[privilege] = expire_timestamp

        if privilege == PRIVILEGES["PrivPublishStream"]:
            self.privileges[PRIVILEGES["privPublishVideoStream"]] = expire_timestamp
            self.privileges[PRIVILEGES["privPublishAudioStream"]] = expire_timestamp
            self.privileges[PRIVILEGES["privPublishDataStream"]] = expire_timestamp

    def expire_time(self, expire_timestamp):
        self.expire_at = expire_timestamp

    def pack_msg(self):
        buf = ByteBuf()
        buf.put_uint32(self.nonce)
        buf.put_uint32(self.issued_at)
        buf.put_uint32(self.expire_at)
        buf.put_string(self.room_id)
        buf.put_string(self.user_id)
        buf.put_tree_map_uint32(self.privileges)
        return buf.pack()

    def serialize(self):
        msg = self.pack_msg()
        # HMAC-SHA256 签名
        signature = hmac.new(
            self.app_key.encode('utf-8'),
            msg,
            hashlib.sha256
        ).digest()

        content = ByteBuf().put_bytes(msg).put_bytes(signature).pack()
        return VERSION + self.app_id + base64.b64encode(content).decode('utf-8')


if __name__ == "__main__":
    # 使用你自己的 app_id / app_key 替换
    token = AccessToken(
        app_id="your_app_id",
        app_key="your_app_key",
        room_id="room123",
        user_id="user456",
    )
    # 有效期，例如 12 小时后过期（时间戳，单位秒）
    expire_ts = int(time.time()) + 12 * 3600
    token.add_privilege(PRIVILEGES["PrivPublishStream"], expire_ts)
    token.add_privilege(PRIVILEGES["PrivSubscribeStream"], expire_ts)
    token.expire_time(expire_ts)

    print(token.serialize())