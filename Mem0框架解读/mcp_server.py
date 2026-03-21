"""
轻量 MCP 记忆服务（FastAPI + FastMCP）。

功能概览：
1) 暴露 MCP 工具接口（新增记忆、检索记忆）
2) 通过 HTTP 挂载在 FastAPI 下（/mcp）
3) 使用 mem0 的 Memory 客户端读写向量记忆

说明：
- 这是“开发态友好配置”，例如 allowed_hosts/origins 全放开。
- 生产环境请收敛安全策略与敏感配置来源（建议改为环境变量）。
"""

import json
import contextlib

import uvicorn
from fastapi import FastAPI
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.server import TransportSecuritySettings

from mem0 import Memory

# 创建 MCP 服务实例：
# - stateless_http=True：每次请求独立处理
# - json_response=True：框架按 JSON 响应处理工具结果
# - streamable_http_path="/": streamable-http 根路径
memory_mcp = FastMCP(
    "mem0-mcp",  # MCP 服务名（给这个 MCP 实例起的标识名）
    stateless_http=True,  # 使用无状态 HTTP：每次请求独立，不依赖服务端长期会话状态
    json_response=True,  # 工具返回值按 JSON 响应处理（通常建议返回 dict/list）
    streamable_http_path="/",  # Streamable HTTP 的挂载路径（这里是根路径）
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,  # 关闭 DNS Rebinding 防护（开发方便，但生产环境不推荐）
        allowed_hosts=["*"],  # 允许所有 Host 头访问（开发方便，生产应收敛到白名单域名/IP）
        allowed_origins=["*"],  # 允许所有跨域来源（CORS 全放开；生产建议限定前端域名）
    ),
)

# 生成可挂载到 FastAPI 的 MCP ASGI app。
memory_server = memory_mcp.streamable_http_app()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 生命周期钩子。

    在服务启动时进入 MCP session_manager 的上下文，
    在服务关闭时自动退出并清理会话相关资源。
    """
    # AsyncExitStack 便于集中管理多个异步上下文（当前只用到 1 个）。
    async with contextlib.AsyncExitStack() as stack:
        # 启动 MCP 会话管理器，确保工具调用期间会话能力可用。
        await stack.enter_async_context(memory_mcp.session_manager.run())
        # yield 之前是“启动阶段”，yield 之后进入“对外服务阶段”。
        yield


# 主 FastAPI 应用，挂载 MCP 子应用到 /mcp。
app = FastAPI(title="Memory Service", lifespan=lifespan)
app.mount("/mcp", memory_server)

# mem0 客户端配置：
# - embedder: 负责向量化
# - vector_store: 负责向量存储与检索
# - llm: 负责对话/推理模型调用
config = {
    "embedder": {
        "provider": "openai",
        "config": {
            "api_key": "EMPTY",
            "openai_base_url": "http://localhost:5001/v1",
            "model": "custom_embedding",
        },
    },
    # 向量库配置
    "vector_store": {
        "provider": "upstash_vector",
        "config": {
            "url": "xxx",  # TODO: 注意替换，upstash创建索引时会给
            "token": "xxx",
            "collection_name": "memory_test2",  # upstash 的 namespace
            "enable_embeddings": False,  #  是否使用upstash内置的embedding模型
        }
    },
    "llm": {
        "provider": "vllm",
        "config": {
            "model": "qwen25",  # 对应你网关里用的 model 名
            "vllm_base_url": "http://localhost:5002/v1", # 注意：这里是 base_url，不要带 /chat/completions
            "api_key": "EMPTY",
        },
    },
}


def _id_from_request(ctx: Context) -> tuple[str, str]:
    """Read uid directly from query string or X-UID header."""
    # 从 MCP 上下文中取出底层 HTTP request。
    request = ctx.request_context.request
    if request is None:
        raise ValueError("Missing request context")

    # 用户标识：优先 query 参数 uid，其次请求头 x-uid。
    uid = request.query_params.get("uid") or request.headers.get("x-uid")
    if not uid:
        raise ValueError("Missing uid. Use /<service>?uid=<your_uid> or send X-UID header")
    # 运行标识（可选）：用于同一用户下进一步区分会话/任务。
    rid = request.query_params.get("rid", None) or request.headers.get("x-rid", None)

    return uid, rid


@memory_mcp.tool()
async def add_memory(text: str, ctx: Context | None = None) -> str:
    """
    新增记忆：当用户表达关于自身的信息、偏好、长期事实，或明确要求“记住某件事”时调用。

    Args:
        text (str): 要写入记忆的文本内容。
    """
    # 从请求上下文提取 user_id / run_id。
    uid, rid = _id_from_request(ctx)
    # 由统一 config 初始化 Memory 客户端（当前每次调用都会创建一次实例）。
    memory = Memory.from_config(config)
    # 写入记忆。
    result = memory.add(messages=text, user_id=uid, run_id=rid)
    return json.dumps(result, ensure_ascii=False)


@memory_mcp.tool()
async def search_memory(text: str, ctx: Context | None = None,) -> str:
    """
    检索记忆：在回答用户问题前优先调用，用于补充历史上下文与个性化信息。

    Args:
        text (str): 检索查询文本。
    """
    # 从请求上下文提取 user_id / run_id。
    uid, rid = _id_from_request(ctx)
    # 由统一 config 初始化 Memory 客户端（当前每次调用都会创建一次实例）。
    memory = Memory.from_config(config)
    result = memory.search(query=text, user_id=uid, run_id=rid)
    return json.dumps(result, ensure_ascii=False)


if __name__ == "__main__":
    # 本地直接运行入口：启动 FastAPI 服务。
    uvicorn.run("mcp_server:app", host="0.0.0.0", port=8080)