# encoding: utf-8
'''
@author: huaiyu-LI
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: lee1012116@126.com
@software: pycharm
@file: main.py
@time: 18-12-17 下午10:11
@desc:
'''

from SpamEmail import spamEmailWords
import re

# spam 类对象
spam = spamEmailWords()
# 保存词频词典
spamDict = {}
normDict = {}
testDict = {}

# 保存每封邮件中出现的词
wordList = []
wordsDict = {}

# 保存预测结果，key为文件名，值为预测类别
testResult = {}
# 分别获得正常邮件、垃圾邮件以及测试文件
normFileList = spam.get_file_list('./data/normal')
spamFileList = spam.get_file_list('./data/spam')
testFileLsit = spam.get_file_list('./data/test')

# 获取训练集中正常邮件与垃圾邮件的数量
normFilelen = len(normFileList)
spamFilelen = len(spamFileList)
# 获得停用词表，用于对停用词过滤
stopList = spam.getStopWords()

# 获得正常邮件中的词频
for filename in normFileList:
    wordList.clear()
    for line in open('./data/normal/' + filename, 'r', encoding='gbk'):
        # 过滤掉非中文字符
        rule = re.compile(r"[^\u4e00-\u9fa5]")
        line = rule.sub("", line)
        # 将每封邮件出现的词保存在wordList中
        spam.get_word_list(line, wordList, stopList)
    # 统计每个词在所有邮件中出现的次数
    spam.add_to_dict(wordList, wordsDict)
normDict = wordsDict.copy()

# 获得垃圾邮件中的词频
wordsDict.clear()
for filename in spamFileList:
    wordList.clear()
    for line in open('./data/spam/' + filename, 'r', encoding='gbk'):
        rule = re.compile(r"[^\u4e00-\u9fa5]")
        line = rule.sub("", line)
        spam.get_word_list(line, wordList, stopList)
    spam.add_to_dict(wordList, wordsDict)
spamDict = wordsDict.copy()

# 测试邮件
for filename in testFileLsit:
    testDict.clear()
    wordsDict.clear()
    wordList.clear()
    for line in open('./data/test/' + filename, 'r', encoding='gbk'):
        rule = re.compile(r"[^\u4e00-\u9fa5]")
        line = rule.sub('', line)
        spam.get_word_list(line, wordList, stopList)
    spam.add_to_dict(wordList, wordsDict)
    testDict = wordsDict.copy()
    # 通过计算每个文件中频p（s|w）来得到对分类影响最大的15个词
    wordProList = spam.get_test_words(testDict, spamDict, normDict, normFilelen, spamFilelen)
    # 对每封邮件得到的15个词计算贝叶斯概率
    p = spam.calBayes(wordProList, spamDict, normDict)
    if p > 0.9:
        testResult.setdefault(filename, 1)
    else:
        testResult.setdefault(filename, 0)

# 计算分类准确率(测试集中文件低于1000的为正常邮件)
testAccuracy = spam.calAccuracy(testResult)
for i, ic in testResult.items():
    print(i + "/" + str(ic))
print(testAccuracy)
