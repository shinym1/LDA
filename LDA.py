import jieba
import os
import collections
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
import math
import numpy as np
import gensim
import csv
import pandas as pd
from sklearn.model_selection import KFold
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def produce_pun(path):
    '''生成标点符号列表'''
    with open(os.path.join(path, 'cn_punctuation.txt'), 'r', encoding='utf-8', errors='ignore') as f:
        punction = f.read()
    punction = punction.replace('\n', '')
    return punction


def produce_stop(path):
    '''停用词列表'''
    with open(os.path.join(path, 'cn_stopwords.txt'), 'r', encoding='utf-8', errors='ignore') as f:
        stop = f.read()
    stop = stop.replace('\n', '')
    return stop


def read_txt(file):
    '''读取原始文本'''
    with open(file, 'r', encoding='gbk', errors='ignore') as f:
        r_txt = f.read()
    r_txt = r_txt.replace('本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com',
                          '')
    return r_txt


def remove_word(r_txt, punction):
    '''生成以词为单元的列表，并去除标点符号或者停用词'''
    txt_words = []
    txt_words_len = 0
    for words in jieba.cut(r_txt):
        if (words not in punction) and (not words.isspace()):
            txt_words.append(words)
            txt_words_len += 1
    return txt_words, txt_words_len


def remove_char(r_txt, punction):
    '''生成以字为单元的列表，并去除标点符号或者停用词'''
    txt_words = []
    txt_words_len = 0
    for words in jieba.cut(r_txt):
        if (words not in punction) and (not words.isspace()):
            for char in words:
                txt_words.append(char)
                txt_words_len += 1
    return txt_words, txt_words_len


def single_file_word(path, file):
    txt_words = []
    punction = produce_stop(path)
    r_txt = read_txt(file)
    tem_txt_words, txt_words_len = remove_word(r_txt, punction=punction)
    return tem_txt_words, txt_words_len


def single_file_char(path, file):
    txt_words = []
    punction = produce_stop(path)
    r_txt = read_txt(file)
    tem_txt_words, txt_words_len = remove_char(r_txt, punction=punction)
    return tem_txt_words, txt_words_len


def produce_char(path):
    txt_words = []
    punction = produce_pun(path)
    for file in os.listdir(os.path.join(path, 'txt')):
        r_txt = read_txt(os.path.join(path, 'txt', file))
        tem_txt_words, txt_words_len = remove_char(r_txt, punction=punction)
        txt_words.extend(tem_txt_words)
    return txt_words


def produce_word(path):
    txt_words = []
    punction = produce_pun(path)
    for file in os.listdir(os.path.join(path, 'txt')):
        r_txt = read_txt(os.path.join(path, 'txt', file))
        tem_txt_words, txt_words_len = remove_word(r_txt, punction=punction)
        txt_words.extend(tem_txt_words)
    return txt_words


def split_list(lst, chunk_size, chunk_num, step):
    '''生成tokens'''
    # for i in range(0, len(lst), chunk_size):
    #     if i < chunk_num*chunk_size:
    #         return lst[i:i + chunk_size]
    return [lst[i:i+chunk_size] for i in range(0, len(lst), step) if i < chunk_num*step]


path = './novel_set'
# file = './novel_set/txt/白马啸西风.txt'
infer = ['碧血剑']
s_txt = []
idx = {}
index = 0
for i, name in enumerate(infer):
    file = os.path.join(path, 'txt', name+'.txt')
    # txt_words, _ = single_file_word(path, file)
    txt_words, _ = single_file_word(path, file)
    split_txt = split_list(txt_words, 100, 1000, 100)
    idx[name] = range(index, index+len(split_txt))
    index = index + len(split_txt)
    s_txt = s_txt + split_txt

print(len(s_txt))

dictionary = corpora.Dictionary(s_txt)
# 将文档转换为词袋表示
corpus_bow = [dictionary.doc2bow(doc) for doc in s_txt]

# 创建 KFold 对象，指定折数 k
k = 10
kf = KFold(n_splits=k, shuffle=True)

# 进行 k 折交叉验证
fold = 1
perplexity_list = []


for train_index, test_index in kf.split(corpus_bow):
    # print(f"Fold: {fold}")
    # print("Train Index:", train_index)
    # print("Test Index:", test_index)
    X_train, X_test = [corpus_bow[i] for i in train_index], [corpus_bow[i] for i in test_index]
    lda_model = gensim.models.LdaModel(X_train, num_topics=10, id2word=dictionary, passes=20)
    # 打印主题-词分布
    # topics = lda_model.print_topics(num_words=3)
    perplexity = lda_model.log_perplexity(X_test) #对数空间困惑度
    print(perplexity)
    perplexity_list.append(perplexity)
    fold += 1



print(perplexity_list)


perplexity_list_word = [-22.40, -19.00, -11.74, -10.42, -9.21]
perplexity_list_char = [-13.14, -9.49, -7.43, -6.98, -6.63]
# 数据量比较大，需要一定时间
# 验证Zipf's law
# produce_hemi(path)  #calculate entropy


# path = './'

# 数据和标签
# channels = ['通道 0', '通道 1', '通道 3', '通道 6', '通道 7']
# f_scores = [6.86, 6.81, 1.89, 3.48, 5.92]
#
# # 创建柱状图
# plt.figure(figsize=(8, 6))
# plt.rcParams['font.sans-serif'] = ['SimSun']
# plt.rcParams['font.size'] = 14
# plt.bar(channels, f_scores, width=0.4)
# for i, v in enumerate(f_scores):
#     plt.text(i, v, str(round(v, 1)), ha='center', va='bottom')

# 添加标题和标签
# plt.title('各通道信号的F值分数')
# plt.xlabel('信号通道')
# plt.ylabel('F值')
# plt.savefig('f.png', dpi=600)
# # 显示图表
# plt.show()



# # 创建词袋模型
# dictionary = corpora.Dictionary(corpus)
# print(dictionary)
# # 将文档转换为词袋表示
# corpus_bow = [dictionary.doc2bow(doc) for doc in corpus]
# print(corpus_bow)
# # 构建LDA模型
# lda_model = gensim.models.LdaModel(corpus_bow, num_topics=2, id2word=dictionary, passes=10)
#
# # 打印主题-词分布
# topics = lda_model.print_topics(num_words=3)
# for topic in topics:
#     print(topic)
#
# out = lda_model.get_document_topics([(0, 1), (2, 1)])
# print(out)