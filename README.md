长篇小说生成Agent系统
主模型 : DeepSeek (deepseek-chat，V4发布后替换模型名即可)
嵌入模型: 千问 text-embedding-v3 (DashScope API)
向量存储: FAISS
前端   : Streamlit

安装依赖:
  pip install streamlit faiss-cpu numpy requests

运行:
  streamlit run agent.py
