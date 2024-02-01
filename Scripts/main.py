import json
import os
import csv
import time

from tqdm import tqdm
from aliyunsdkalinlp.request.v20200629 import GetSaChGeneralRequest
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.acs_exception.exceptions import ClientException
from aliyunsdkcore.acs_exception.exceptions import ServerException

from aip import AipNlp

import  emoji

# 这里在使用时将等号右边的内容全部删除，并改为由英文引号""包围的key和secret
# 这里为阿里云填写处
access_key_id = os.environ['NLP_AK_ENV']
access_key_secret = os.environ['NLP_SK_ENV']
# 创建AcsClient实例(阿里云)
clientAli = AcsClient(
    access_key_id,
    access_key_secret,
    "cn-hangzhou"
)

# 这里为百度开放平台填写处
APP_ID = '输入APP_ID'
API_KEY = '输入API_KEY'
SECRET_KEY = '输入SECRET_KEY'
# 创建百度实例
clientBaidu = AipNlp(APP_ID, API_KEY, SECRET_KEY)


# 这里的函数可以用于处理单条或者错误
def SentimentAnalysisAli(text,emojitreat):
    request = GetSaChGeneralRequest.GetSaChGeneralRequest()
    # 添加emoji表情处理方式，使用emoji python包
    if emojitreat == 'replace':
        text = emoji.replace_emoji(string=text, replace=lambda chars, data_dict: data_dict['zh'])
    elif emojitreat == 'delete':
        text = emoji.replace_emoji(string=text, replace='')
    else:
        print('可能的emoji处理方式拼写错误')
    request.set_Text(text)
    request.set_ServiceCode("alinlp")
    response = clientAli.do_action_with_exception(request)
    resp_obj = json.loads(response)
    return resp_obj


def SentimentAnalysisBaidu(text,emojitreat):
    # 添加emoji表情处理方式，使用emoji python包
    if emojitreat == 'replace':
        text = emoji.replace_emoji(string=text, replace=lambda chars, data_dict: data_dict['zh'])
    elif emojitreat == 'delete':
        text = emoji.replace_emoji(string=text, replace='')
    else:
        print('可能的emoji处理方式拼写错误')
    processed_text = text.lstrip('\ufeff').replace(':', '')
    gbk_encoded_text = processed_text.encode('utf-8').decode('utf-8')
    resp_obj = clientBaidu.sentimentClassify(gbk_encoded_text)
    return resp_obj


# 这里提供了整体处理
def StartAnalysis(input_file, output_file, colnum=1, Ali=True, Baidu=True,emojitreat='replace'):
    colnum = int(colnum)
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline='',
                                                                 encoding='utf-8') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        if Ali == True and Baidu == True:
            # 写入新文件的标题行
            writer.writerow(
                ['text', 'AliSentiment', 'AliPositive_prob', 'AliNeutral_prob', 'AliNegative_prob', 'AliRequestId',
                 'BaiduSentiment', 'BaiduConfidence', 'BaiduPositive_prob', 'BaiduNegative_prob', 'BaiduLogid'])
            for row in tqdm(reader, desc="Processing"):
                text = row[colnum - 1]  # 假设文本在第一列
                resultAli = SentimentAnalysisAli(text,emojitreat)
                resultBaidu = SentimentAnalysisBaidu(text,emojitreat)
                # 百度QPS为2,需要缓一下
                time.sleep(0.6)
                # 对于阿里进行提取
                if 'Data' in resultAli:
                    # 解析 Data 字段
                    dataAli = json.loads(resultAli['Data'])['result']
                    updated_row = [text, dataAli['sentiment'], dataAli['positive_prob'], dataAli['neutral_prob'],
                                   dataAli['negative_prob'], resultAli['RequestId']]
                else:
                    # 如果没有 Data 字段，整个返回内容写入第二列
                    updated_row = [text, json.dumps(resultAli), None, None, None, None]

                # 对百度进行提取
                if 'items' in resultBaidu:
                    # 解析items字段
                    dataBaidu = resultBaidu['items'][0]
                    # 替换原数字表述
                    if dataBaidu['sentiment'] == 1:
                        dataBaidu['sentiment'] = '中性'
                    elif dataBaidu['sentiment'] == 0:
                        dataBaidu['sentiment'] = '负向'
                    elif dataBaidu['sentiment'] == 2:
                        dataBaidu['sentiment'] = '正向'
                    # 拓展行
                    updated_row.extend([dataBaidu['sentiment'], dataBaidu['confidence'], dataBaidu['positive_prob'],
                                        dataBaidu['negative_prob'], resultBaidu['log_id']])
                else:
                    # 如果没有item字段，则整个返回内容写入第七列
                    updated_row.extend([json.dumps(resultBaidu), None, None, None, None])
                # 写入处理后的数据到新文件
                writer.writerow(updated_row)
        elif Ali == True and Baidu != True:
            # 写入新文件的标题行
            writer.writerow(
                ['text', 'AliSentiment', 'AliPositive_prob', 'AliNeutral_prob', 'AliNegative_prob', 'AliRequestId'])
            for row in tqdm(reader, desc="Processing"):
                text = row[colnum - 1]  # 假设文本在第一列
                resultAli = SentimentAnalysisAli(text,emojitreat)
                # 如果提示并发限制则开启下面的代码
                # time.sleep(0.6)
                # 对于阿里进行提取
                if 'Data' in resultAli:
                    # 解析 Data 字段
                    dataAli = json.loads(resultAli['Data'])['result']
                    updated_row = [text, dataAli['sentiment'], dataAli['positive_prob'], dataAli['neutral_prob'],
                                   dataAli['negative_prob'], resultAli['RequestId']]
                else:
                    # 如果没有 Data 字段，整个返回内容写入第二列
                    updated_row = [text, json.dumps(resultAli), None, None, None, None]
                # 写入处理后的数据到新文件
                writer.writerow(updated_row)
        elif Ali != True and Baidu == True:
            # 写入新文件的标题行
            writer.writerow(
                ['text', 'BaiduSentiment', 'BaiduConfidence', 'BaiduPositive_prob', 'BaiduNegative_prob', 'BaiduLogid'])
            for row in tqdm(reader, desc="Processing"):
                text = row[colnum - 1]  # 假设文本在第一列
                resultBaidu = SentimentAnalysisBaidu(text,emojitreat)
                # 百度QPS为2,需要缓一下
                time.sleep(0.6)
                # 进行占位以对其结果
                updated_row = [text]

                # 对百度进行提取
                if 'items' in resultBaidu:
                    # 解析items字段
                    dataBaidu = resultBaidu['items'][0]
                    # 替换原数字表述
                    if dataBaidu['sentiment'] == 1:
                        dataBaidu['sentiment'] = '中性'
                    elif dataBaidu['sentiment'] == 0:
                        dataBaidu['sentiment'] = '负向'
                    elif dataBaidu['sentiment'] == 2:
                        dataBaidu['sentiment'] = '正向'
                    # 拓展行
                    updated_row.extend([dataBaidu['sentiment'], dataBaidu['confidence'], dataBaidu['positive_prob'],
                                        dataBaidu['negative_prob'], resultBaidu['log_id']])
                else:
                    # 如果没有item字段，则整个返回内容写入第七列
                    updated_row.extend([json.dumps(resultBaidu), None, None, None, None])
                # 写入处理后的数据到新文件
                writer.writerow(updated_row)
        else:
            print('参数输入错误，请至少选择一个分析平台（阿里或百度）')


if __name__ == "__main__":
    # 进行整体处理时使用下面这条函数
    StartAnalysis(input_file='../TestFiles/TestFIle.csv', output_file='../Results/TestResult.csv', colnum=1, Ali=True,
                  Baidu=True,emojitreat='replace')

    # 进行单条处理时使用
    # print(SentimentAnalysisAli(text='这是一个测试样例',emojitreat='replace'))
    # print(SentimentAnalysisBaidu(text='这是一个测试样例',emojitreat='replace'))
