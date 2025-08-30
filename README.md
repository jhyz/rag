Run: uvicorn main:app --port 7866
# 本地RAG解决方案

这是一个基于LlamaIndex的本地RAG（检索增强生成）问答系统。

## 功能特性
- 支持上传结构化和非结构化数据
- 可以创建多个知识库
- 基于向量检索的问答功能
- 支持DeepSeek大模型

## 环境准备

### 安装依赖
```bash
pip install -r requirements.txt
```

### 设置环境变量
1. 复制`.env.example`文件并重命名为`.env`
2. 在`.env`文件中填入您的API密钥：
   ```
   DEEPSEEK_API_KEY="your_deepseek_api_key_here"
   ```

## 运行项目
```bash
python main.py
```

然后访问 http://127.0.0.1:7866

## 使用流程
1. 首先上传数据（结构化或非结构化）
2. 创建知识库
3. 在RAG问答界面进行提问

## 注意事项
- 项目依赖于DeepSeek API，请确保您有有效的API密钥
- 上传的文件将保存在`File`目录
- 生成的向量数据库将保存在`VectorStore`目录
- 敏感信息已从代码中移除，请通过环境变量配置API密钥