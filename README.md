# 使用阿里云服务进行情感分析

该项目使用阿里云的[情感分析服务](https://help.aliyun.com/document_detail/179345.html)进行情感分析。相较于自行撰写代码使用API进行情感分析，该脚本提供了一个开箱即用（进行简单配置），进行批量情感分析的工具。

## 在开始之前

首先，您需要获得使用阿里云服务的App Key和App Secret 。 如果您是杨老师课题组的成员或杨老师的学生。请您联络组会群中头像为一只大象在白色背景中的成员，或者直接联络杨老师获取配置的App Ke与App Secret。

在运行之前，请找到脚本（Scripts文件夹下的main.py文件），并修改下面的内容：

如果您没有将App Key和App Secret添加至环境变量，则将下面的内容：
```python
access_key_id = os.environ['NLP_AK_ENV']
access_key_secret = os.environ['NLP_SK_ENV']
```
修改为：

```python
access_key_id = '这里替换为你获得的AppKey'
access_key_secret = '这里替换为你获得的AppSecret'
```
**请注意** 请不要将AppKey或AppSecret分享给其他任何人或上传至网络


其次，您需要正确安装运行脚本所需的包。如果您使用的是Pycharm，则您在加载本项目时应当会有提示进行自动安装，否则，您需要手动安装，您可以在终端中分别运行下面的代码来安装：

```
pip install aliyun-python-sdk-alinlp
pip install tqdm
```

## 开始使用

### StartAnalysis函数

`StartAnalysis()`函数包含两个参数，一个为传入的CSV文件（包含需要分析的文本内容），另一个为分析结果的输出地址。在代码中以`TestFile.csv`为例（包含20条不同情感的文本）。在运行代码时，如无特殊需要，可以直接修改并保存`TestFile.csv`的内容并直接运行，分析结果将会被保存至`Results`文件夹下。

**请注意**

需要分析的文本内容需要放置在一个`utf-8`编码的`CSV`文件的第一列内。否则程序会报错或无法分析。

### SentimentAnalysis函数

`SentimentAnalysis()`函数可以用于分析单条文本并获得结果，用于处理批量分析时因网络或其他问题获得的错误结果。通常情况下，如果未能获得正确的结果，将在结果文件的第二列中呈现报错原因或代码。通常情况下，请检查网络连接或文本是否超出了1000个字。如果手动复制粘贴内容仍然获得报错代码，则可以访问[这个页面](https://help.aliyun.com/document_detail/179345.html)查看错误码部分。或者联系*组会群中背景为白色，主体为一直坐着的大象的同学*。

## 解读结果

我们关注的内容主要集中在下面的内容：

 - `sentiment`：为判断的情感类型，包含正面、负面、中性三种情况
 - `positive_prob`：句子极性为正面的概率大小，0-1之间，保留4位小数
 - `neutral_prob`：句子极性为中性的概率大小，0-1之间，保留4位小数
 - `negative_prob`：句子极性为负面的概率大小，0-1之间，保留4位小数
 - `RequestId`：唯一请求id，排查问题的依据

## 引用

您可以考虑下面的引用格式

阿里云. (2024, January 24). 自然语言处理NLP. https://ai.aliyun.com/nlp

或

aliyun. (2024, January 24). Natural language processing. https://ai.aliyun.com/nlp
