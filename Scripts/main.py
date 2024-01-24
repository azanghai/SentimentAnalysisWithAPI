import json
import os
import csv
from tqdm import tqdm
from aliyunsdkalinlp.request.v20200629 import GetSaChGeneralRequest
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.acs_exception.exceptions import ClientException
from aliyunsdkcore.acs_exception.exceptions import ServerException

# 这里在使用时将等号右边的内容全部删除，并改为由英文引号""包围的key和secret
access_key_id = os.environ['NLP_AK_ENV']
access_key_secret = os.environ['NLP_SK_ENV']

# 创建AcsClient实例
client = AcsClient(
    access_key_id,
    access_key_secret,
    "cn-hangzhou"
)
# 这里的函数可以用于处理单条或者错误
def SentimentAnalysis(text):
    request = GetSaChGeneralRequest.GetSaChGeneralRequest()
    request.set_Text(text)
    request.set_ServiceCode("alinlp")
    response = client.do_action_with_exception(request)
    resp_obj = json.loads(response)
    return resp_obj

# 这里提供了整体处理
def StartAnalysis(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # 写入新文件的标题行
        writer.writerow(['text', 'sentiment', 'positive_prob', 'neutral_prob', 'negative_prob', 'RequestId'])

        for row in tqdm(reader, desc="Processing"):
            text = row[0]  # 假设文本在第一列
            result = SentimentAnalysis(text)

            if 'Data' in result:
                # 解析 Data 字段
                data = json.loads(result['Data'])['result']
                updated_row = [text, data['sentiment'], data['positive_prob'], data['neutral_prob'], data['negative_prob'], result['RequestId']]
            else:
                # 如果没有 Data 字段，整个返回内容写入第二列
                updated_row = [text, json.dumps(result), None, None, None, None]

            # 写入处理后的数据到新文件
            writer.writerow(updated_row)



if __name__ == "__main__":
    # 进行整体处理时使用下面这条函数
    StartAnalysis(input_file='../TestFiles/TestFIle.csv',output_file='../Results/TestResult.csv')

    # 进行单条处理时使用
    # print(SentimentAnalysis('这里是一个示例'))