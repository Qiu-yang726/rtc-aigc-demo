"""
rag_llm_server/main.py

一体化服务：火山引擎实时音视频 AIGC + 第三方大模型（豆包 doubao-seed-2-1-turbo-260628）。

对外提供 3 类接口：
1. POST /getScenes
       获取场景列表并自动生成 RTC Token（「获取 token」接口）。
2. POST /proxy?Action=StartVoiceChat | StopVoiceChat
       代理火山引擎 AIGC OpenAPI 请求，打开 / 关闭语音对话服务（「打开火山引擎服务」接口）。
3. POST /v1/chat/completions  （别名：POST /chat/completions）
       接入第三方大模型（CustomLLM）：接收 RTC 透传的 messages，
       转发到火山方舟（ARK）调用豆包 doubao-seed-2-1-turbo-260628，并以 SSE 流式返回。

第三方大模型接入方式（在 scenes/Custom.json 的 LLMConfig 中配置）：
    "LLMConfig": {
      "Mode": "CustomLLM",                                              // 必填，固定值
      "Url": "https://modulator-carnivore-droplet.ngrok-free.dev/v1/chat/completions",  // 必填，本服务公网地址
      "APIKey": "",                                                     // 可选，鉴权 Token
      "ModelName": "doubao-seed-2-1-turbo-260628"                       // 透传到请求体的 model
    }
"""
import os
import uuid
import time
import json
import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from config import settings
from services.token_builder import AccessToken, PRIVILEGES
from services.utils import read_files, assert_val, response_wrapper, Signer
from services.llm_services import llm_service

app = FastAPI(title="RTC Third-party LLM Service (Doubao)")

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 读取场景配置（scenes 目录，与官方 Demo 保持一致）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCENES = read_files(os.path.join(BASE_DIR, "scenes"), ".json")

# SSE 响应头：禁用缓存/缓冲，保证流式输出即时到达 RTC
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


# =====================================================================
# 接口三：接入第三方大模型（CustomLLM）—— 转发到火山方舟调用豆包
# =====================================================================

def _verify_api_key(request: Request):
    """若配置了 CUSTOM_LLM_API_KEY，则校验请求头 Authorization: Bearer <key>。"""
    expected = settings.CUSTOM_LLM_API_KEY
    if not expected:
        return
    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {expected}":
        raise PermissionError("Invalid API key")


async def _chat_response(body: dict):
    """统一的对话处理：根据请求体中的 stream 字段决定流式或非流式返回。"""
    if not body.get("stream", False):
        result = await llm_service.chat(body)
        return JSONResponse(content=result)

    return StreamingResponse(
        llm_service.stream_chat(body),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@app.get("/")
async def index():
    return {
        "service": "rtc-third-party-llm",
        "model": llm_service.model,
        "status": "ok",
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model": llm_service.model}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI 兼容接口：RTC 通过此接口请求豆包模型（默认 SSE 流式返回）。"""
    _verify_api_key(request)
    body = await request.json()
    return await _chat_response(body)


@app.post("/chat/completions")
async def chat_completions_alias(request: Request):
    """别名接口：兼容将 Url 配置为 /chat/completions 的场景。"""
    _verify_api_key(request)
    body = await request.json()
    return await _chat_response(body)


# =====================================================================
# 接口二：代理 AIGC OpenAPI 请求 —— 打开 / 关闭火山引擎语音对话服务
# =====================================================================

@app.post("/proxy")
async def proxy(request: Request):
    """
    代理 AIGC 的 OpenAPI 请求（StartVoiceChat / StopVoiceChat）
    """
    action = request.query_params.get("Action")
    version = request.query_params.get("Version", "2024-12-01")

    try:
        body_data = await request.json()
    except Exception:
        body_data = {}

    async def logic():
        assert_val(action, "Action 不能为空")
        assert_val(version, "Version 不能为空")

        scene_id = body_data.get("SceneID")
        assert_val(scene_id, "SceneID 不能为空, SceneID 用于指定场景的 JSON")

        json_data = SCENES.get(scene_id)
        assert_val(json_data, f"{scene_id} 不存在, 请先在 scenes 目录下定义该场景的 JSON.")

        voice_chat = json_data.get("VoiceChat", {})
        account_config = json_data.get("AccountConfig", {})

        assert_val(account_config.get("accessKeyId"), "AccountConfig.accessKeyId 不能为空")
        assert_val(account_config.get("secretKey"), "AccountConfig.secretKey 不能为空")

        request_body = {}
        if action == "StartVoiceChat":
            request_body = voice_chat
        elif action == "StopVoiceChat":
            app_id = voice_chat.get("AppId")
            room_id = voice_chat.get("RoomId")
            task_id = voice_chat.get("TaskId")

            assert_val(app_id, "VoiceChat.AppId 不能为空")
            assert_val(room_id, "VoiceChat.RoomId 不能为空")
            assert_val(task_id, "VoiceChat.TaskId 不能为空")

            request_body = {
                "AppId": app_id,
                "RoomId": room_id,
                "TaskId": task_id,
            }

        # 构造并签名请求
        host = "rtc.volcengineapi.com"
        open_api_request_data = {
            "method": "POST",
            "path": "/",
            "params": {"Action": action, "Version": version},
            "headers": {
                "Host": host,
                "Content-Type": "application/json",
            },
            "body": request_body,
        }

        signer = Signer(open_api_request_data, "rtc")
        signer.add_authorization(account_config)

        # 用 separators=(',', ':') 序列化 body，与签名时保持完全一致
        body_bytes = json.dumps(request_body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

        # 发起真实请求
        url = f"https://{host}/"
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url,
                params={"Action": action, "Version": version},
                headers=open_api_request_data["headers"],
                content=body_bytes,
                timeout=30.0,
            )
            return resp.json()

    return await response_wrapper("proxy", logic, contain_metadata=False)


# =====================================================================
# 接口一：获取场景列表并自动生成 RTC Token
# =====================================================================

@app.post("/getScenes")
async def get_scenes(request: Request):
    """
    获取场景列表并自动生成 Token
    """

    async def logic():
        result_scenes = []
        for key, data in SCENES.items():
            scene_config = data.get("SceneConfig", {})
            rtc_config = data.get("RTCConfig", {})
            voice_chat = data.get("VoiceChat", {})

            app_id = rtc_config.get("AppId")
            room_id = rtc_config.get("RoomId")
            user_id = rtc_config.get("UserId")
            token = rtc_config.get("Token")
            app_key = rtc_config.get("AppKey")

            assert_val(app_id, f"{key} 场景的 RTCConfig.AppId 不能为空")

            # 自动生成 Token 逻辑
            if app_id and (not token or not user_id or not room_id):
                new_room_id = room_id or str(uuid.uuid4())
                new_user_id = user_id or str(uuid.uuid4())

                rtc_config["RoomId"] = new_room_id
                voice_chat["RoomId"] = new_room_id

                rtc_config["UserId"] = new_user_id
                if voice_chat.get("AgentConfig") and isinstance(voice_chat["AgentConfig"].get("TargetUserId"), list):
                    voice_chat["AgentConfig"]["TargetUserId"][0] = new_user_id

                assert_val(app_key, f"自动生成 Token 时, {key} 场景的 AppKey 不可为空")

                token_builder = AccessToken(app_id, app_key, new_room_id, new_user_id)
                token_builder.add_privilege(PRIVILEGES["PrivSubscribeStream"], 0)
                token_builder.add_privilege(PRIVILEGES["PrivPublishStream"], 0)
                token_builder.expire_time(int(time.time()) + (24 * 3600))

                rtc_config["Token"] = token_builder.serialize()

            # 构造前端所需的 SceneConfig
            scene_config["id"] = key
            scene_config["botName"] = voice_chat.get("AgentConfig", {}).get("UserId")

            interrupt_mode = voice_chat.get("Config", {}).get("InterruptMode")
            scene_config["isInterruptMode"] = (interrupt_mode == 0)

            llm_config = voice_chat.get("Config", {}).get("LLMConfig", {})
            vision_config = llm_config.get("VisionConfig", {})
            scene_config["isVision"] = vision_config.get("Enable")

            snapshot_config = vision_config.get("SnapshotConfig", {})
            scene_config["isScreenMode"] = (snapshot_config.get("StreamType") == 1)

            avatar_config = voice_chat.get("Config", {}).get("AvatarConfig", {})
            scene_config["isAvatarScene"] = avatar_config.get("Enabled")
            scene_config["avatarBgUrl"] = avatar_config.get("BackgroundUrl")

            # 移除敏感的 AppKey
            rtc_config_safe = rtc_config.copy()
            rtc_config_safe.pop("AppKey", None)

            result_scenes.append({
                "scene": scene_config,
                "rtc": rtc_config_safe,
            })

        return {"scenes": result_scenes}

    return await response_wrapper("getScenes", logic)


if __name__ == "__main__":
    print(f"RTC Third-party LLM Server (Doubao) is running at http://localhost:{settings.PORT}")
    uvicorn.run("main:app", host="0.0.0.0", port=settings.PORT, reload=True)