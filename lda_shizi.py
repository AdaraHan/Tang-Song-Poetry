# -*- coding: utf-8 -*-

import codecs  # 编码转换模块
import pyLDAvis.gensim
from gensim import corpora
from gensim.models import LdaModel

input_path = 'shizi_qss.txt'
lda_path = 'lda_shizi_qss.html'

train = []

fp = codecs.open(input_path, 'r', encoding='utf-8')  # 打开分词结果文件
# 将分词结果转换为列表形式
for line in fp:
    if line != '':
        line = line.split()
        train.append([w for w in line])

dictionary = corpora.Dictionary(train)  # 构建词典
corpus = [dictionary.doc2bow(text) for text in train]  # 转换数据：训练时使用的数据需要使用词典转换数据
lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=10, passes=100)  # 训练模型：num_topics为主题数量，passes为训练次数

for topic in lda.print_topics(num_words=10):  # num_words：每个主题下输出的term的数目
    termNumber = topic[0]
    print(topic[0], ':', sep='')
    listOfTerms = topic[1].split('+')
    for term in listOfTerms:
        listItems = term.split('*')
        print(' ', listItems[1], '(', listItems[0], ')', sep='')

# 可视化
d = pyLDAvis.gensim.prepare(lda, corpus, dictionary, mds='mmds')  # lda:计算好的话题模型；corpus：文档词频矩阵；dictionary：词语空间
pyLDAvis.show(d)  # 展示在浏览器
pyLDAvis.save_html(d, lda_path)  # 将结果保存为该html文件
