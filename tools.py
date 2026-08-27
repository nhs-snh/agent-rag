"""
FusionRAG 自定义业务工具
==========================
定义 Agent 可调用的业务工具（Tool Calling）。

工具注册机制：
  LangChain 的 @tool 装饰器会自动：
  1. 提取函数签名和 docstring
  2. 生成 JSON Schema（描述入参类型和含义）
  3. 注册到 Agent 的工具列表中

  当用户提问时，LLM 根据工具描述决定是否调用，
  并生成符合 JSON Schema 的参数，由 LangChain 自动解析并执行。

实际生产中，这些工具会调用真实的后端 API（订单系统、物流系统等）。
这里用 mock 数据演示工具调用链路。
"""

from langchain_core.tools import tool


@tool
def query_order_status(order_id: str) -> str:
    """查询订单状态。当用户询问某个订单的当前状态、是否发货、是否完成时使用。
    需要用户提供订单号。

    Args:
        order_id: 订单号，例如 "ORD-20240001"
    """
    # ===== Mock 数据（实际项目中这里调用后端 API）=====
    mock_orders = {
        "ORD-20240001": {
            "status": "已发货",
            "logistics_company": "顺丰速运",
            "tracking_number": "SF1234567890",
            "estimated_delivery": "2024-12-25",
        },
        "ORD-20240002": {
            "status": "待发货",
            "payment_status": "已付款",
            "expected_ship_date": "2024-12-23",
        },
        "ORD-20240003": {
            "status": "已完成",
            "delivery_date": "2024-12-20",
            "review_status": "未评价",
        },
    }

    result = mock_orders.get(order_id)
    if result:
        return f"订单 {order_id} 状态：{result}"
    else:
        return f"未找到订单 {order_id}，请确认订单号是否正确。"


@tool
def query_logistics(tracking_number: str) -> str:
    """查询物流追踪信息。当用户想知道包裹到哪了、物流进度、快递状态时使用。

    Args:
        tracking_number: 快递单号，例如 "SF1234567890"
    """
    # ===== Mock 数据 =====
    mock_logistics = {
        "SF1234567890": {
            "company": "顺丰速运",
            "current_status": "运输中",
            "latest_update": "2024-12-24 10:30 到达【杭州转运中心】",
            "history": [
                "2024-12-23 14:00 已揽收（深圳）",
                "2024-12-23 22:00 到达【深圳转运中心】",
                "2024-12-24 10:30 到达【杭州转运中心】",
            ],
        },
    }

    result = mock_logistics.get(tracking_number)
    if result:
        history_text = "\n".join(result["history"])
        return (
            f"快递单号：{tracking_number}\n"
            f"快递公司：{result['company']}\n"
            f"当前状态：{result['current_status']}\n"
            f"物流轨迹：\n{history_text}"
        )
    else:
        return f"未找到快递单号 {tracking_number} 的物流信息。"


@tool
def submit_refund(order_id: str, reason: str) -> str:
    """提交退款申请。当用户要求退货退款、申请售后时使用。

    Args:
        order_id: 需要退款的订单号
        reason: 退款原因，例如 "商品质量问题" 或 "不想要了"
    """
    # ===== Mock 数据 =====
    return (
        f"退款申请已提交：\n"
        f"订单号：{order_id}\n"
        f"退款原因：{reason}\n"
        f"申请状态：审核中（预计 1-3 个工作日）\n"
        f"退款金额将在审核通过后原路退回。"
    )


@tool
def query_refund_policy() -> str:
    """查询退款退货政策。当用户问"能退吗""退货运费谁出""退款要多久"等
    关于退款退货规则的问题时使用。不需要任何参数。
    """
    return (
        "退款退货政策：\n"
        "1. 7天无理由退货：签收后7天内可申请无理由退货，商品需完好未使用。\n"
        "2. 质量问题退货：商品存在质量问题，15天内可申请退货，运费由卖家承担。\n"
        "3. 退款时效：退款申请审核通过后，3-5个工作日内原路退回。\n"
        "4. 退货运费：无理由退货由买家承担运费，质量问题由卖家承担。\n"
        "5. 不支持退货的商品：定制品、贴身衣物、食品等特殊品类。"
    )


# 所有工具的注册列表，供 Agent 初始化时使用
ALL_TOOLS = [
    query_order_status,
    query_logistics,
    submit_refund,
    query_refund_policy,
]
