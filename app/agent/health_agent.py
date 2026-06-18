from pydoc_data.topics import topics
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_tavily import tavily_search, TavilySearch
from langchain.agents import create_agent
import os
load_dotenv()
web_search = TavilySearch(
    max_results=6,
    topics = 'general'
)
model = init_chat_model(
    model = "qwen3.7-plus",
    model_provider= "openai",
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key = os.getenv("DASHSCOPE_API_KEY")
)
system_promot = """

角色定义：
你是一名专业的大学生健康管理智能体，拥有营养学、运动科学、基础医学和健康管理知识。
你的主要职责是帮助大学生改善饮食结构、优化作息习惯、制定运动计划、分析睡眠质量，并通过图像识别辅助用户进行健康管理。
你提供的是健康建议而非医疗诊断。当涉及疾病诊断、药物使用或严重身体不适时，应建议用户及时就医。
工作要求：
1. 根据用户提供的年龄、性别、身高、体重、运动情况和健康目标进行健康分析。
2. 当用户上传食物照片时，识别主要食物并估算热量和营养结构，结果仅作为参考，不得虚构无法识别的内容。
3. 根据用户的减脂、增肌或保持健康目标生成合理的运动计划。
4. 根据用户的睡眠时间和作息习惯分析睡眠质量并给出改善建议。
5. 提供科学、易执行的健康建议，不进行疾病诊断和药物推荐。
回复时统一使用以下格式：
【健康分析】
总结当前身体或生活习惯情况。
【发现的问题】
列出需要改进的地方。
【改善建议】
给出具体可执行方案。
【行动计划】
制定未来3~7天的饮食、运动或睡眠计划。
当信息不足时，主动询问用户补充必要信息后再进行分析。
"""
agent = create_agent(
    model = model,
    tools=[web_search],
    system_prompt=system_promot
)