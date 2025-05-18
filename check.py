import os
from ragas.metrics import faithfulness,answer_relevancy,context_recall,context_precision,answer_similarity,answer_correctness
from datasets import Dataset
from ragas import evaluate, RunConfig
from langchain_community.llms.tongyi import Tongyi
from langchain_community.embeddings.dashscope import DashScopeEmbeddings

os.environ["DASHSCOPE_API_KEY"] = "sk-1a6a96ed75ad4bb0899ecec9b57863ba"
llm = Tongyi(model_name="qwen-turbo")
embeddings=DashScopeEmbeddings()

data_samples = {
    'question': [
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
    ],
    'answer': [
        '江苏省总面积10.72万平方千米',
        '江苏有8526万人',
        '2023年江苏GDP达12.82万亿元',
        '南京是江苏省省会',
        '江苏现有26家国家5A级旅游景区',
        '历史名人有徐霞客、顾炎武、周恩来等',
        '有盐水鸭、阳澄湖大闸蟹、扬州炒饭等',
        '共有16所双一流建设高校',
        '昆曲、评弹、扬剧等',
        '主要属于江淮官话和吴方言',
        '截至2023年底高铁里程达2504公里'
    ],
    'contexts': [
        ['江苏省总面积10.72万平方千米，下辖13个设区市...'],  # 移除多余嵌套
        ['2024年末江苏省户籍人口8526万人...'],
        ['2023年江苏省地区生产总值达12.82万亿元，增长5.8%...'],
        ['南京市是江苏省省会，副省级城市...'],
        ['江苏现有26家国家5A级旅游景区，数量全国第一...'],
        ['江苏历史名人：徐霞客（地理学家）、顾炎武（思想家）、周恩来（开国总理）...'],
        ['南京盐水鸭、苏州阳澄湖大闸蟹、扬州炒饭...'],
        ['江苏省有南京大学、东南大学等16所双一流高校...'],
        ['昆曲（人类非遗）、苏州评弹、扬州扬剧...'],
        ['江苏省方言主要分为江淮官话区（北部）和吴方言区（南部）...'],
        ['江苏高铁运营里程达2504公里，居全国前列...']
    ],
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

dataset = Dataset.from_dict(data_samples)
run_config = RunConfig(
    max_retries=5,
    max_wait=120,
    log_tenacity=True
)
metrics=[faithfulness,
         answer_relevancy,
         context_recall,
         context_precision,
         answer_similarity,
         answer_correctness
         ]
print("开始计算。。。。。")
score = evaluate(dataset,metrics,
                 llm=llm,embeddings=embeddings,
                 raise_exceptions=False,run_config=run_config)
score.to_pandas().to_csv('D:/work/python/localrag/check_ans/check_ans.csv', sep=',')
print("计算完成。。。。。。")
print(score)

# 评估结果
#{'faithfulness': 1.0000, 'answer_relevancy': 0.6457, 'context_recall': 1.0000, 'context_precision': 1.0000, 'answer_similarity': 0.9262, 'answer_correctness': 0.9816}