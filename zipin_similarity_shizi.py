"""
需求：使用TF-IDF算法计算《全唐诗》《全宋诗》实字文档的相似度
步骤：1. 将两文档转换为向量矩阵（无需分词，都是实字）；2. 计算TF-IDF矩阵；3.计算余弦相似度
参考：https://blog.csdn.net/yjh_SE007/article/details/108429694?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-108429694-blog-113918690.pc_relevant_landingrelevant&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-108429694-blog-113918690.pc_relevant_landingrelevant&utm_relevant_index=2
"""
# -*- coding: utf-8 -*-
import math
import numpy as np

text_sum = 2

s = ''
with open("shizi_qts.txt", 'r', encoding='utf-8') as f:
    while True:
        line = f.readline().strip()
        s += line
        if not line:
            break
print('s=', s)

s2 = []
for word in s:
    if word != '\n':
        s2 += word

print('s2=', s2)
text1 = s2
print('text1=', text1)

z = ''
with open("shizi_qss.txt", 'r', encoding='utf-8') as f:
    while True:
        line = f.readline().strip()
        z += line
        if not line:
            break

print('z=', z)
z2 = []
for word in z:
    if word != '\n':
        z2 += word

print('z2=', z2)
text2 = z2
print('text2=', text2)

print("-----------------------------------创建词汇------------------------------------")
vocabulary = []
vocabulary = text1 + text2
vocabulary = list(set(vocabulary))
print('vocabulary=', vocabulary)
print("-----------------------------------创建文本的向量矩阵:start---------------------------------------")
# 创建文本1的向量矩阵
arr1 = []
for t in vocabulary:
    if text1.count(t):
        arr1.append(text1.count(t))
    else:
        arr1.append(0)
print('arr1=', arr1)
# 创建文本2的向量矩阵
arr2 = []
for t in vocabulary:
    if text2.count(t):
        arr2.append(text2.count(t))
    else:
        arr2.append(0)
print('arr2=', arr2)
print("-----------------------------创建文本的向量矩阵:end------------------------------------")
print('len(vocabulary)=', len(vocabulary))
print('len(arr1)=', len(arr1))
print('len(arr2)=', len(arr2))
print("-----------------------------TF:start------------------------------------")


# 计算词频TF
def compute_tf(list_words):
    tf_list = []
    for i in list_words:
        tf_list.append(i / len(list_words))
    return tf_list


arr1_tf = compute_tf(arr1)
print('arr1_tf=', arr1_tf)

arr2_tf = compute_tf(arr2)
print('arr2_tf=', arr2_tf)
print("-----------------------------TF:end------------------------------------")

print("-----------------------------IDF:start------------------------------------")


# 计算词语出现在文档的次数
def count_words(text1, text2):
    text_conut_arr = [0] * len(vocabulary)
    # print(text_conut_arr)
    # count=0
    # for i in range(0,len(text)):
    #     if text[i].
    for i in range(0, len(vocabulary)):
        # print(vocabulary[i])
        if vocabulary[i] in text1:
            text_conut_arr[i] += 1
            if vocabulary[i] in text2:
                text_conut_arr[i] += 1
    return text_conut_arr


# 文档一词语出现在文档数的向量
c1 = count_words(text1, text2)
print('c1=', c1)
# 文档二词语出现在文档数的向量
c2 = count_words(text2, text1)
print('c2=', c2)


# 计算逆向文件频率:IDF
def file_idf(c1):
    idf_arr1 = []
    for i in c1:
        idf_arr1.append(math.log(text_sum / (i + 1)))
    return idf_arr1


# 计算逆向文件频率:IDF
def file_idf(c2):
    idf_arr2 = []
    for i in c2:
        idf_arr2.append(math.log(text_sum / (i + 1)))
    return idf_arr2


arr1_idf = file_idf(c1)
print('arr1_idf=', arr1_idf)
arr2_idf = file_idf(c2)
print('arr2_idf=', arr2_idf)
print("-----------------------------IDF:end------------------------------------")

print("---------------------------------计算TF-IDF的向量矩阵:start-----------------------------------------")


# print(arr1_tf)
# print(arr1_idf)
# 计算TF-IDF的向量矩阵
def tf_idf(arr_tf, arr_idf):
    tfidf_arr = []
    for i in arr_tf:
        for j in arr_idf:
            tfidf_arr.append(i * j)
    return tfidf_arr


arr1_tfidf = tf_idf(arr1_tf, arr1_idf)
print(arr1_tfidf)
arr2_tfidf = tf_idf(arr2_tf, arr2_idf)
print(arr2_tfidf)
print("---------------------------------计算TF-IDF的向量矩阵:end-----------------------------------------")

print("----------------------------余弦相似度--------------------------------")


# 余弦相似度
def cosine_similarity(x, y, norm=False):
    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = [0] * len(y)
    if x == zero_list or y == zero_list:
        return float(1) if x == y else float(0)
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(y))])
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
    return 0.5 * cos + 0.5 if norm else cos


similarity = cosine_similarity(arr1_tfidf, arr2_tfidf)
print("这两篇文档的相似度为：{:%}".format(similarity))
print(similarity)
