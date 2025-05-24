import os
import traceback
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel  # 添加这行导入
from anthropic import Client
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferMemory
from datetime import datetime
import asyncio
from pyppeteer import launch
from urllib.parse import quote
import logging
from playwright.async_api import async_playwright  # 新增


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 配置常量
ANTHROPIC_API_KEY = "sk-ant-api03-NigxhOOf5hG5OERSjRRwC-cYw-ThEtfQAA"
# CHROMIUM_PATH = '/Users/xsser/Downloads/chrome-mac 3/Chromium.app/Contents/MacOS/Chromium'
# 修改Chrome路径配置
CHROME_PATHS = [
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
    '/usr/lib/chromium/chromium',
    '/usr/lib/chromium-browser/chromium-browser'
]

DEFAULT_TIMEOUT = 30000

class HoroscopeRequest(BaseModel):
    year: str
    month: str
    day: str
    hour: str
    sex: str
    name: str
    question: str


async def get_horoscope_info(year: str, month: str, day: str, hour: str, sex: str, name: str) -> str:
    """获取命盘信息"""
    try:
        logger.info("初始化 Playwright...")
        async with async_playwright() as p:
            # 启动浏览器
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage'
                ]
            )

            logger.info("浏览器启动成功")

            # 创建新的上下文和页面
            context = await browser.new_context()
            page = await context.new_page()

            try:
                # 构造URL
                date_str = f"{year}{month.zfill(2)}{day.zfill(2)}"
                gender = "female" if sex == "1" else "male"
                time_period = int(hour)
                encoded_name = quote(name)
                url = f"https://ziwei.pub/astrolabe/?d={date_str}&t={time_period}&leap=false&g={gender}&type=solar&n={encoded_name}"

                logger.info(f"访问URL: {url}")

                # 访问页面并等待加载完成
                await page.goto(url, wait_until='networkidle')

                # 等待特定元素出现
                logger.info("等待命盘元素加载...")
                await page.wait_for_selector('.iztro-astrolabe', timeout=60000)

                # 等待额外的时间确保渲染完成
                await asyncio.sleep(2)

                # 获取内容
                content = await page.locator('.iztro-astrolabe').text_content()

                if not content:
                    raise ValueError("未能提取到命盘内容")

                logger.info("成功获取命盘内容")
                return content.strip()

            finally:
                # 确保资源被正确释放
                await context.close()
                await browser.close()

    except Exception as e:
        logger.error(f"获取命盘信息失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={
                "message": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc()
            }
        )


def create_analysis_chain():
    """创建分析链"""
    try:
        llm = ChatAnthropic(
            anthropic_api_key=ANTHROPIC_API_KEY,
            model='claude-3-5-sonnet-20241022',
            temperature=0,
            max_tokens=4000
        )

        # 验证 API key
        try:
            # 简单的测试请求
            response = llm.invoke("测试连接")
            logger.info("API 连接测试成功")
        except Exception as e:
            logger.error(f"API 连接测试失败: {str(e)}")
            raise ValueError(f"API key 验证失败: {str(e)}")

        memory = ConversationBufferMemory(input_key="input", memory_key="chat_history")

        prompt_template = """
# System Prompt: 高级紫微斗数分析系统

你是一位精通紫微斗数文墨天机方法的专业命理师，擅长多维度分析和实用性建议。请基于以下信息使用文墨天机进行深入解析，每个方面的分析字数不少于200字，总字数不少于500字。
必须使用文墨天机方法测算，并且使用文墨天机命理师开头作为你的个人介绍。
## INPUT

### 命盘基础信息 
{horoscope_info}


### 用户咨询问题 
{input}

## OUTPUT STRUCTURE

### 1. 命盘总体分析
#### 1.1 核心格局评估
- 命局格局等级
- 主星品质评估
- 格局特征解读
- 整体能量场评估

#### 1.2 先天禀赋解析
- 性格特质分析
- 思维模式特点
- 行为倾向评估
- 潜在天赋识别

#### 1.3 核心命盘特征
- 命主紫微主星状态
- 命宫所在宫位特点
- 三方四正格局分析
- 年月日时四柱特征

### 2. 星耀组合深度分析
#### 2.1 主星分布与作用
- 紫微命系诸星分析
- 天府命系诸星分析
- 主星落宫特征评估
- 主星间互动关系

#### 2.2 辅星配置解析
- 吉星分布特点
- 煞星影响评估
- 辅星组合效果
- 特殊星耀作用

#### 2.3 四化信息剖析
- 四化星分布状况
- 四化作用效果
- 四化串宫影响
- 化气变迁规律

### 3. 宫位全息解读
#### 3.1 命宫系统
- 命宫星耀组合
- 迁移宫互动
- 疾厄宫影响
- 相貌宫特征

#### 3.2 事业系统
- 官禄宫特点
- 迁移宫机遇
- 财帛宫状态
- 福德宫支持

#### 3.3 感情系统
- 夫妻宫特征
- 子女宫状态
- 兄弟宫互动
- 父母宫影响

#### 3.4 健康系统
- 命宫健康特征
- 疾厄宫警示
- 福德宫调节
- 身心平衡状态

### 4. 大运流年分析
#### 4.1 大运评估
- 当前大运特征
- 大运星耀作用
- 与本命互动关系
- 大运机遇挑战

#### 4.2 流年解析
- 流年总体特征
- 流年吉凶判定
- 流年关键时点
- 流年注意事项

#### 4.3 流月流日分析
- 近期流月特点
- 关键日期提醒
- 时间点选择
- 调节建议

### 5. 多维度运势分析
#### 5.1 事业发展
- 职业发展方向
- 晋升机会点
- 潜在风险点
- 团队协作状况

#### 5.2 财运分析
- 财源特点分析
- 理财方向建议
- 投资机会评估
- 破财风险提醒

#### 5.3 感情婚姻
- 感情发展特点
- 婚姻质量评估
- 桃花机遇分析
- 感情建议指导

#### 5.4 健康状况
- 体质特点分析
- 易感疾病提醒
- 保健要点建议
- 调养方案推荐

#### 5.5 学习进修
- 学习能力评估
- 知识领域建议
- 技能发展方向
- 学习方法推荐

### 6. 具体问题解答
#### 6.1 问题分析
- 问题根源探讨
- 相关星耀分析
- 影响因素评估
- 发展趋势预判

#### 6.2 解决方案
- 近期行动建议
- 中期发展规划
- 长期布局思路
- 具体执行步骤

### 7. 开运化解建议
#### 7.1 趋吉避凶指导
- 有利时间选择
- 有利方位建议
- 有利行业领域
- 注意规避事项

#### 7.2 能量提升方案
- 修身养性建议
- 习惯养成指导
- 能量场调节
- 自我提升方法

#### 7.3 具体行动指南
- 短期行动计划（1-3个月）
- 中期发展规划（3-12个月）
- 长期目标设定（1-3年）
- 执行要点提醒

### 8. 开运建议汇总
#### 8.1 核心建议要点
- 近期重点事项
- 关键行动建议
- 特别注意事项
- 发展机遇提示

#### 8.2 配套行动方案
- 具体执行步骤
- 时间节点安排
- 预期效果评估
- 调整优化建议

## 分析原则

1. 内容要求：
- 分析全面且系统
- 重点突出关键信息
- 建议具体可执行
- 预期效果明确

2. 表达方式：
- 专业术语解释清晰
- 分析逻辑严谨
- 建议实用易懂
- 语言积极正向


## 补充说明

1. 分析深度：
- 从表象到本质
- 从现象到根源
- 从问题到方案
- 从理论到实践

2. 建议特点：
- 针对性强
- 可操作性高
- 时效性明确
- 灵活性足够

3. 效果评估：
- 短期效果可见
- 中期成效可期
- 长期价值持久
- 整体收益明显
"""

        prompt = PromptTemplate(
            input_variables=["horoscope_info", "input"],
            template=prompt_template
        )

        return LLMChain(
            llm=llm,
            prompt=prompt,
            memory=memory,
            verbose=True
        )

    except Exception as e:
        logger.error(f"创建分析链失败: {str(e)}")
        logger.error(traceback.format_exc())
        raise


# 创建 FastAPI 应用
app = FastAPI(
    title="紫微斗数分析系统",
    version="1.0",
    description="基于在线命盘与AI的紫微斗数分析系统"
)

# 配置模板
templates = Jinja2Templates(directory="templates")
os.makedirs("templates", exist_ok=True)

# 创建分析链
chain = create_analysis_chain()


@app.get("/ziweidoushu2", response_class=HTMLResponse)
async def read_root(request: Request):
    """返回主页"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze")
async def analyze_horoscope(request: HoroscopeRequest):
    """分析紫微斗数命盘"""
    try:
        # 获取命盘信息
        horoscope_info = await get_horoscope_info(
            request.year,
            request.month,
            request.day,
            request.hour,
            request.sex,
            request.name
        )
        print(f'性别{request.sex}')
        print(horoscope_info)
        if not horoscope_info:
            raise HTTPException(status_code=500, detail="命盘信息获取失败")
        print(request.question,horoscope_info)
        # 使用AI进行分析
        result = chain.predict(
            input=request.question,
            horoscope_info=horoscope_info
        )

        return {"result": result}

    except Exception as e:
        logger.error(f"分析过程出错: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """系统健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
