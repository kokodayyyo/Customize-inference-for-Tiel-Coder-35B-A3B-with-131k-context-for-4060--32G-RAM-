"""网络客户端构造：访问本机回环时**必须关闭代理自动检测**。

踩坑记录
--------
本机装了 Clash / v2ray 之类的系统代理（WinINET 里设成 ``127.0.0.1:7897``）时，
httpx 的 ``trust_env=True``（默认）会通过 ``urllib.request.getproxies()``
**从 Windows 注册表**读到这个代理，然后把发往 ``127.0.0.1`` 的请求也丢给代理，
代理返回 **502**。

而 PowerShell、浏览器、``urllib`` 都会遵守 Windows 的"绕过本地地址"
（ProxyOverride 里的 ``<local>``），所以症状表现为：

    同一 URL、同一时刻：
      httpx  -> HTTP 502   （3.5 秒）
      urllib -> HTTP 200   （0.03 秒）

看起来像"只有 Python 连不上本机服务"，实际上是被自己的代理拦了。

后果不止健康检查：网关照样子用 httpx 转发给 llama-server，等于**整个 API
全部 502**；``server.start()`` 也会因为轮询不到 /health 而空转直到超时。

因此本项目内所有访问 llama-server（127.0.0.1）的 httpx 客户端统一用
本模块的构造函数，强制 ``trust_env=False``。本地回环流量永远不该走代理。

诊断脚本：``scripts/diag_http.py`` 可以复现并对比各种配置。
"""

from __future__ import annotations

import httpx

# 传给 httpx 的公共参数：不读环境变量/注册表里的代理设置
NO_PROXY_ENV: dict[str, object] = {"trust_env": False}


def local_client(timeout: float = 60.0, **kwargs) -> httpx.Client:
    """访问本机服务的同步客户端（不走代理）。"""
    kwargs.update(NO_PROXY_ENV)
    return httpx.Client(timeout=timeout, **kwargs)


def local_async_client(timeout: float = 60.0, **kwargs) -> httpx.AsyncClient:
    """访问本机服务的异步客户端（不走代理）。"""
    kwargs.update(NO_PROXY_ENV)
    return httpx.AsyncClient(timeout=timeout, **kwargs)


def local_get(url: str, timeout: float = 5.0, **kwargs):
    """对本机地址做一次同步 GET（不走代理）。"""
    return httpx.get(url, timeout=timeout, **NO_PROXY_ENV, **kwargs)


def local_post(url: str, timeout: float = 60.0, **kwargs):
    """对本机地址做一次同步 POST（不走代理）。"""
    return httpx.post(url, timeout=timeout, **NO_PROXY_ENV, **kwargs)


def local_stream(method: str, url: str, timeout: float = 60.0, **kwargs):
    """对本机地址发起流式请求（不走代理）。"""
    return httpx.stream(method, url, timeout=timeout, **NO_PROXY_ENV, **kwargs)
