#前置环境
#encoding = utf-8
import jieba
from jieba.analyse import *
import jieba.posseg as pseg

def keywords(text):



    #词的提取
    with open('sample.txt') as f:
        data = f.read()

    for keyword, weight in extract_tags(data, topK=20, withWeight= True):#文本的关键词，根据出现的次数有权重，但我认为权重还应该是词出现的位置、词与词之间的联系
        print('%s %s' % (keyword, weight))

    for keyword, weight in textrank(data, withWeight = True):#我们应该使用这种办法，因为他会根据两个词之间的位置，建立联系，除了出现次数的排序，还有词云联系绳索数量的排序。
        print('%s %s' % (keyword, weight))

    #判断词性
    p = open(r'1.txt', 'r', encoding = 'gbk')   #这个在python2中语法错误
    q = open(r'2.txt', 'w', encoding = 'gbk') 
    for line in p.readlines():
        words = pseg.cut(line)
        for word, flag in words:
            q.write(str(word) + str(flag) + "  ")
        q.write('\n')