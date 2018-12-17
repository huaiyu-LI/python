# encoding: utf-8
'''
@author: huaiyu-LI
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: lee1012116@126.com
@software: pycharm
@file: SpamEmail.py
@time: 18-12-16 下午2:37
@desc:
'''
import jieba
import os


class spamEmailWords(object):
    # 获得停用词表
    def getStopWords(self):
        stopList = []
        with open('./data/chinese_stopwords.txt', 'r', encoding='gbk') as f:
            for line in f:
                stopList.append(line[:len(line) - 1])
                # print(line[:len(line) - 1])

        # for line in open("./data/中文停用词表.txt", 'rb'):
        # stopList.append(line[:len(line) - 1])
        # print(line[:len(line) - 1].decode('utf-8'))
        return stopList

    # 获得词典
    def get_word_list(self, content, wordsList, stopList):
        # 分词结果放入res_list
        res_list = list(jieba.cut(content))
        for i in res_list:
            if i not in stopList and i.strip() != '' and i != None:
                if i not in wordsList:
                    wordsList.append(i)

    # 若列表中的词已在词典中，则加1,否则添加进词典
    def add_to_dict(self, wordList, wordsDict):
        for item in wordList:
            if item in wordsDict.keys():
                wordsDict[item] += 1
            else:
                # wordList[item]=1
                wordsDict.setdefault(item, 1)

    def get_file_list(self, filePath):
        filenames = os.listdir(filePath)
        return filenames

    # 通过计算每个文件中(s|w)来得到对分类影响最大的15个词
    def get_test_words(self, testDict, spamdict, normDict, normFilelen, spamFilelen):
        wordProbList = {}
        for word, num in testDict.items():
            if word in spamdict.keys() and word in normDict.keys():
                # 该文件中包含词的个数
                pw_s = spamdict[word] / spamFilelen
                pw_n = normDict[word] / normFilelen
                ps_w = pw_s / (pw_s + pw_n)
                wordProbList.setdefault(word, ps_w)
            if word in spamdict.keys() and word not in normDict.keys():
                pw_s = spamdict[word] / spamFilelen
                pw_n = 0.01
                ps_w = pw_s / (pw_s + pw_n)
                wordProbList.setdefault(word, ps_w)
            if word not in spamdict.keys() and word in normDict.keys():
                pw_s = 0.01
                pw_n = normDict[word] / normFilelen
                ps_w = pw_s / (pw_s + pw_n)
                wordProbList.setdefault(word, ps_w)
            if word not in spamdict.keys() and word not in normDict.keys():
                wordProbList.setdefault(word, 0.47)
        wordProbList = sorted(wordProbList.items(), key=lambda d: d[1], reverse=True)[0:15]
        return (wordProbList)

    def calBayes(self, wordList, spamdict, normdict):
        ps_w = 1
        ps_n = 1
        for word, prob in wordList:
            print(word + "/" + str(prob))
            ps_w *= (prob)
            ps_n *= (1 - prob)
        p = ps_w / (ps_w + ps_n)
        return p

    def calAccuracy(self, testResult):
        rightCount = 0
        errorCount = 0
        for name, catagory in testResult.items():
            if (int(name) < 1000 and catagory == 0) or (int(name) > 1000 and catagory == 1):
                rightCount += 1
            else:
                errorCount += 1
            return rightCount / (rightCount + errorCount)

    pass


if __name__ == '__main__':
    spam = spamEmailWords()
    spam.getStopWords()
