import os
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision, answer_similarity, \
    answer_correctness
from datasets import Dataset
from ragas import evaluate, RunConfig
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from chat import get_model_response
from typing import Dict, Any
from langchain_openai import ChatOpenAI

# 初始化环境变量
os.environ["DASHSCOPE_API_KEY"] = "sk-"
os.environ["DEEPSEEK_API_KEY"] = "sk-"

# RAG配置参数
RAG_CONFIG = {
    "model": "deepseek-chat",
    "temperature": 0.5,
    "max_tokens": 1024,
    "history_round": 3,
    "db_name": "default",
    "similarity_threshold": 0.6,
    "chunk_cnt": 5
}

embeddings = DashScopeEmbeddings()


def generate_rag_response(question: str) -> Dict[str, Any]:
    """动态生成RAG响应"""
    mock_input = {"text": question, "files": []}
    history = [[question, ""]]

    try:
        response_gen = get_model_response(
            multi_modal_input=mock_input,
            history=history,
            **RAG_CONFIG
        )

        final_response = None
        for updated_history, contexts in response_gen:
            final_response = {
                "answer": updated_history[-1][1],
                "contexts": [ctx.split('\n') for ctx in contexts.split('##') if ctx]
            }
        return final_response
    except Exception as e:
        print(f"生成响应失败: {str(e)}")
        return {"answer": "", "contexts": []}


# 初始化评估数据集
questions = [
    '江苏面积是多少',
    '江苏有多少人',
    '江苏的GDP是多少',
    '江苏的省会是哪里',
    '江苏有多少个5A级景区',
    '江苏历史上有哪些著名人物',
    '江苏的特色美食有哪些',
    '江苏有多少所双一流高校',
    '江苏的地方戏曲是什么',
    '江苏的方言主要属于什么语系',
    '江苏高铁总里程多少公里'
]

data_samples = {
    'question': questions,
    'answer': [],
    'contexts': [],
    'ground_truth': [
        '江苏省总面积10.72万平方千米',
        '江苏省户籍人口8526万人',
        '12.82万亿元',
        '南京',
        '26个',
        '徐霞客、顾炎武、周恩来',
        '盐水鸭、大闸蟹、扬州炒饭',
        '16所',
        '昆曲、评弹、扬剧',
        '江淮官话和吴方言',
        '2504公里'
    ]
}

# 生成RAG响应
for q in questions:
    response = generate_rag_response(q)
    data_samples['answer'].append(response['answer'])
    data_samples['contexts'].append(response['contexts'])

dataset = Dataset.from_dict(data_samples)

# 配置评估参数
run_config = RunConfig(
    max_retries=10,
    max_wait=300,
    timeout=60,
    log_tenacity=True
)

metrics = [
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_similarity,
    answer_correctness
]

# 执行评估
print("开始计算。。。。。")
score = evaluate(
    dataset,
    metrics,
    embeddings=embeddings,
    raise_exceptions=False,
    run_config=run_config
)

# 保存结果
score.to_pandas().round(4).to_csv(
    'D:/work/python/localrag/check_ans/check_ans.csv',
    index=False,
    float_format='%.4f'
)
print("计算完成。。。。。。")
print(score)