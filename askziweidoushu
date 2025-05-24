import os
from anthropic import Client
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferMemory

# 保持原有的 Anthropic API密钥
anthropic_api_key = ""

def count_tokens(text):
    client = Client(api_key=anthropic_api_key)
    tokens = client.count_tokens(text)
    return tokens


# 指定紫微斗数文档路径
docs_path = "./github_codes/weiboSpider/article/"

# 检查目录是否存在
if not os.path.exists(docs_path):
    print(f'未找到目录: {docs_path}')
    raise FileNotFoundError(f"目录 {docs_path} 不存在")
else:
    print(f'使用目录: {docs_path}')

# 使用DirectoryLoader加载MD文件
loader = DirectoryLoader(
    docs_path,
    glob="**/*.md",
    loader_cls=TextLoader,
    loader_kwargs={'encoding': 'utf-8'}
)

# 加载文档并计算token数量
documents = []
total_tokens = 0
max_tokens = 180000

print("开始加载紫微斗数参考资料...")
for doc in loader.load():
    doc_tokens = count_tokens(str(doc))
    if total_tokens + doc_tokens <= max_tokens:
        documents.append(doc)
        total_tokens += doc_tokens
    else:
        print(f'总token数 {total_tokens} 超过最大限制 {max_tokens}')
        break

print(f"加载的总token数: {total_tokens}")

# 合并所有文档内容
combined_docs = "\n\n".join([str(doc.page_content) for doc in documents])

# 初始化 ChatAnthropic 模型和会话内存
llm = ChatAnthropic(
    anthropic_api_key=anthropic_api_key,
    model='claude-3-5-sonnet-20240620',
    temperature=0,
    max_tokens=4000
)

memory = ConversationBufferMemory(input_key="input", memory_key="chat_history")

# 创建针对紫微斗数命盘分析的提示模板
prompt_template = """
你现在是一个精通紫微斗数的专业命理师。以下是参考资料和命盘信息。

参考资料内容:
{ziwei_docs}

当前命盘信息:
{horoscope_info}

用户问题: {input}

请基于以上命盘信息和参考资料，进行专业的紫微斗数分析。分析时请：
1. 关注命盘的关键特征（主星配置、宫位特点、四化信息等）
2. 结合参考资料解释星曜在各宫的含义
3. 分析流年流月对命盘的影响
4. 给出具体、实用的建议
5. 如遇需要补充的信息，请明确指出

请用中文专业且详细地回答：
"""

prompt = PromptTemplate(
    input_variables=["ziwei_docs", "horoscope_info", "input"],
    template=prompt_template
)

# 创建 LLMChain
llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# 存储命盘信息
horoscope_info = None


def set_horoscope(info):
    global horoscope_info
    horoscope_info = info
    print("命盘信息已保存！")


print("\n紫微斗数命盘分析系统已准备就绪！")
print("首先请输入命盘信息（输入'END'结束）：")

# 读取命盘信息
lines = []
while True:
    line = input()
    if line.strip().upper() == 'END':
        break
    lines.append(line)

# 保存命盘信息
horoscope_info = "\n".join(lines)
print("\n命盘信息已记录！现在可以开始提问了。")
print("输入问题进行分析，输入'quit'结束对话")

# 主问答循环
while True:
    question = input("\n请输入你的问题: ").strip()

    if question.lower() == 'quit':
        print("感谢使用紫微斗数命盘分析系统！")
        break

    if not question:
        print("问题不能为空，请重新输入。")
        continue

    try:
        result = llm_chain.predict(
            input=question,
            ziwei_docs=combined_docs,
            horoscope_info=horoscope_info
        )
        print("\n分析结果：")
        print(result)

    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        print("请重新输入你的问题。")
