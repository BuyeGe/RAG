import os
# 给你的爬虫起个名字，伪装成一个正规的访问者
os.environ["USER_AGENT"] = "Buye_AI_Bot/1.0"

# 引入 LangChain 里的网页加载工具
from langchain_community.document_loaders import WebBaseLoader
# 引入 LangChain 里的文本切分工具
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 引入 HuggingFace 免费开源的 Embedding 模型工具
from langchain_huggingface import HuggingFaceEmbeddings
# 引入 Chroma 向量数据库
from langchain_community.vectorstores import Chroma

print("🚀 第 1 步：开始抓取网页数据...")

# 1. 设定你要爬取的网址（这里用了一篇经典的 AI 博客文章）
url = "https://lilianweng.github.io/posts/2023-06-23-agent/"

# 2. 实例化加载器（这就是你的 Crawler）
loader = WebBaseLoader(url)

# 3. 执行抓取操作
docs = loader.load()

# 4. 打印结果看看
print(f"✅ 抓取成功！一共抓取了 {len(docs)} 个文档。")
print(f"📄 让我们看看前 200 个字长什么样：\n{docs[0].page_content[:200]}")
print("-" * 50)


print("✂️ 第 2 步：开始将长文本切片...")

# 1. 设置切片规则
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# 2. 执行切分操作，把刚才抓取的 docs 放进去切
splits = text_splitter.split_documents(docs)

# 3. 打印切分结果
print(f"✅ 切分完成！这篇文章被切成了 {len(splits)} 个小文字块。")
print(f"📦 我们来看看第一个文字块的内容：\n{splits[0].page_content}")

print("🧠 第 3 步：开始加载本地 AI 翻译官（第一次运行需要下载模型文件，大概几十MB，请耐心等待）...")

# 1. 实例化本地的 Embedding 模型
# 这里我们选用 "all-MiniLM-L6-v2"，它是目前笔记本电脑上跑得最快、效果也极好的轻量级开源模型
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("🧮 模型加载完毕！正在将文字块转换为数字向量，并存入本地数据库...")

# 2. 将之前切好的文字块 (splits) 和 翻译官 (embeddings_model) 一起放进 Chroma 数据库
# persist_directory="./chroma_db" 的意思是，把算好的数据持久化保存到你当前项目的一个新建文件夹里
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings_model,
    persist_directory="./chroma_db"
)

print("🎉 恭喜！所有数据已经成功变成向量，并安全地存入了你本地的 Chroma 数据库中！")
print("快看一眼你 PyCharm 左侧的项目目录，是不是多了一个叫 chroma_db 的文件夹？")