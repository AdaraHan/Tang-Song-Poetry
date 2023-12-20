"""author：Adara 2023/08/12
需求：使用Transformers库中的BERT模型计算两个诗歌库中的诗句是否使用了古文句式
训练集：poem_sanwen_data_train.csv【两列，一列诗歌，一列标签】
测试集：poem_sanwen_data_test.csv[同上]
模型：清华BERT-CCPoem
步骤：
1. 加载人工标注好的csv数据。
2.将数据转换为datasets.Dataset格式。
3.使用BERT-CCPoem的分词器对数据进行分词。
4.将数据分割为训练集和验证集。
5.使用Trainer进行模型的微调。
报错：
(-2147352567, '发生意外。', (0, 'Microsoft Access Database Engine', '标准表达式中数据类型不匹配。', None, 5003071, -2147217913), None)
"""
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import win32com.client

# 1. 加载人工标注好的csv数据
data_path = "poem_sanwen_data_train.csv"
data = pd.read_csv(data_path, encoding='utf-8')
texts = data['poem'].tolist()
labels = data['label'].tolist()

# 2. 转换数据为HuggingFace Dataset格式，与transformers库配合使用时，易操作、更兼容
dataset = Dataset.from_dict({'text': texts, 'label': labels})

# 3. 使用BERT-CCPoem的分词器
tokenizer = BertTokenizer.from_pretrained("D:\\BERT\\BERT_CCPoem_v1\\BERT_CCPoem_v1\\")
encoding = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=256)


# 对Dataset进行分词
def encode_function(example):
    return tokenizer(example['text'], padding='max_length', truncation=True, max_length=256)


encoded_dataset = dataset.map(encode_function, batched=True)

# 4. 分割数据集为训练集和验证集
split_dataset = encoded_dataset.train_test_split(test_size=0.1)

# 5. 微调BERT-CCPoem模型
model = BertForSequenceClassification.from_pretrained("D:\\BERT\\BERT_CCPoem_v1\\BERT_CCPoem_v1\\", num_labels=2)
training_args = TrainingArguments(
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy="steps",
    save_strategy="epoch",
    output_dir="./bert_poem_model",
    eval_steps=100  # 每100步评估一次
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset['train'],
    eval_dataset=split_dataset['test'],
)

trainer.train()


# 6. 使用模型进行预测
# 从Access数据库中读取要预测的诗歌
def read_poems_from_access(database_path, table_name):
    conn_str = (
            r"Provider=Microsoft.ACE.OLEDB.12.0;"
            r"Data Source=" + database_path + ";"
    )

    conn = win32com.client.Dispatch("ADODB.Connection")
    conn.Open(conn_str)

    rs = win32com.client.Dispatch("ADODB.Recordset")
    rs.Open("[" + table_name + "]", conn, 1, 3)  # 1 adOpenKeyset, 3 adLockOptimistic

    poems = []
    while not rs.EOF:
        # 以列名'诗文'读取诗歌【此处需与数据表中字段一致】
        poem = rs.Fields.Item("诗文").Value
        poems.append(poem)
        rs.MoveNext()

    rs.Close()
    conn.Close()

    return poems


# 批量预测诗歌
def predict_poems_batch(poems, batch_size=32):
    predictions = []

    for i in range(0, len(poems), batch_size):
        batch = poems[i:i + batch_size]
        with torch.no_grad():
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=256)
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_labels = torch.argmax(probs, dim=-1).tolist()
            predictions.extend(pred_labels)

    return predictions


# 从Access数据库中读取诗歌
database_path = 'D:\\Document\\03-Academic Research\\QTS&QSS\\QTS&QSS.accdb'
table_name = 't_qss_new'
new_poems = read_poems_from_access(database_path, table_name)

# 使用批量预测的函数
predicted_labels = predict_poems_batch(new_poems)
print(predicted_labels)

# 将预测结果写入Access数据库
# 创建数据库连接
conn_str = r"Provider=Microsoft.ACE.OLEDB.12.0;Data Source=" + database_path + ";Persist Security Info=False;"
conn = win32com.client.Dispatch('ADODB.Connection')
conn.Open(conn_str)

for index, prediction in enumerate(predicted_labels):
    # 使用index + 1作为ID值，并确保prediction为整数值
    prediction_int = int(prediction)
    query = f"UPDATE {table_name} SET is_sanwen={prediction_int} WHERE ID={index + 1}"
    try:
        conn.Execute(query)
    except Exception as e:
        print(f"Error executing query: {query}")
        print(e)
        # 打印出详细的错误描述，有助于进一步诊断问题
        print("Prediction:", prediction)
        print("ID:", index + 1)

conn.Close()

#  # 使用BERT进行预测，返回预测结果【========非批量，每次只处理一个诗歌，效率低=========】
# def predict_poems(poems):
#     predictions = []
#
#     for poem in poems:
#         with torch.no_grad():
#             inputs = tokenizer(poem, return_tensors='pt', padding=True, truncation=True, max_length=256)
#             outputs = model(**inputs)
#             probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
#             pred_label = torch.argmax(probs, dim=-1).item()
#             predictions.append(pred_label)
#
#     return predictions
#
#
# # 从Access数据库中读取诗歌
# database_path = 'D:\\Document\\03-Academic Research\\QTS&QSS\\QTS&QSS.accdb'
# table_name = 't_qts_new'
# new_poems = read_poems_from_access(database_path, table_name)
#
# # 使用BERT进行预测，返回预测结果
# predicted_labels = predict_poems(new_poems)
# print(predicted_labels)
#
# # 将预测结果写入Access数据库
# # 创建数据库连接
# conn_str = r"Provider=Microsoft.ACE.OLEDB.12.0;Data Source=" + database_path + ";Persist Security Info=False;"
# conn = win32com.client.Dispatch('ADODB.Connection')
# conn.Open(conn_str)
# rs = win32com.client.Dispatch('ADODB.Recordset')
#
# for index, prediction in enumerate(predicted_labels):
#     print('index=', index, '\nprediction=', prediction)
#     # 使用index + 1作为ID值
#     query = f"UPDATE {table_name} SET is_sanwen={prediction} WHERE ID={index + 1}"
#     try:
#         conn.Execute(query)
#     except Exception as e:
#         print(f"Error executing query: {query}")
#         print(e)
#
# conn.Close()
