# 引入必要的工具
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

print("🔍 正在启动本地 AI 检索引擎...")

# 1. 叫醒我们的“翻译官”（必须和之前存数据时用的是同一个模型）
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. 连接到我们刚才建好的本地数据库！
# 注意：这里不需要再爬网页了，直接读取 chroma_db 文件夹
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings_model)

print("✅ 数据库连接成功！\n")

# 3. 输入你想问的问题（你可以随便改成别的英文问题）
question = "What is an AI agent?"
print(f"👤 你的问题: {question}")
print("🧠 正在庞大的数据库中寻找答案...\n")

# 4. 核心科技：相似度搜索 (Similarity Search)
# k=2 的意思是，只返回最相关、最核心的 2 个文字块
docs = vectorstore.similarity_search(question, k=2)

# 5. 打印出找到的答案
for i, doc in enumerate(docs):
    print(f"👇 --- 找到的第 {i+1} 个最相关段落 --- 👇")
    print(doc.page_content)
    print("-" * 50 + "\n")