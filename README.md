# 使用API服务进行情感分析

目前，该脚本可以使用[阿里云]((https://help.aliyun.com/document_detail/179345.html))与[百度开放平台](https://ai.baidu.com/tech/nlp_apply/sentiment_classify)进行批量或单独的情感分析。相较于自行撰写代码使用API进行情感分析，该脚本提供了一个开箱即用（进行简单配置），进行批量情感分析的工具。

## 在开始之前

首先，您需要获得使用阿里云服务的App Key和App Secret或百度云的App Key和App Secret，这取决于您想使用哪一种情感分析服务。***好消息是***如果您是*杨老师课题组*的成员或*杨老师*的学生。请您联络组会群中头像为一只大象在白色背景中的成员，或者直接联络杨老师获取配置的App
Key与App Secret。

该脚本的撰写和测试使用Pyhon 3.10 版本

### 安装运行脚本所需要的库

您需要正确安装运行脚本所需的包。如果您使用的是Pycharm，则您在加载本项目时应当会有提示进行自动安装，否则，您需要手动安装，您可以在终端中分别运行下面的代码来安装：

```
pip install aliyun-python-sdk-alinlp
pip install tqdm
pip install baidu-aip
pip install chardet
```

在运行之前，请找到脚本（Scripts文件夹下的main.py文件），并修改下面的内容：

### 使用阿里云的情感分析服务配置

如果您没有将**阿里云**App Key和App Secret添加至环境变量，则将下面的内容：

```
access_key_id = os.environ[`NLP_AK_ENV`]
access_key_secret = os.environ[`NLP_SK_ENV`]
```

修改为：

```
access_key_id = `这里替换为你获得的AppKey`
access_key_secret = `这里替换为你获得的AppSecret`
```

### 使用百度云的情感分析配置

请在`Scripts`文件夹下`main.py`文件中找到下面的代码处
```
# 这里为百度开放平台填写处
APP_ID = `输入APP_ID`
API_KEY = `输入API_KEY`
SECRET_KEY = `输入SECRET_KEY`
```

并将等号右侧的部分修改为你获得的对应内容（**需要使用英文单引号包围**）

**请注意** 请不要将AppKey或AppSecret分享给其他任何人或上传至网络

## 开始使用

请划至脚本的最后，即`if __name__ == "__main__":`下方的代码进行修改或运行。

### StartAnalysis函数

`StartAnalysis()`
函数包含五个参数，分别为：

 - `input_file`即传入的CSV文件路径和文件名。默认分析文件为`TestFiles`文件夹下的`TestFile.csv`（包含20条不同情感的文本）。在运行代码时，如无特殊需要，可以直接修改并保存`TestFile.csv`的内容并直接运行，分析结果将会被保存至下一个参数指定的地址。
 - `output_file`即分析结果的输出地址。在代码中默认保存在`Results`文件夹下的`TestResult.csv`文件。如果您不为保存分析结果的文件改名，则每次会覆写之前的内容；
 - `colnum` 用于指定数据所在传入文件的哪一列，默认为第一列，即int类型的数字1；
 - `Ali`是否阿里云进行分析，默认为启用（`True`）。如不使用可以修改为`False`；
 - `Baidu`是否使用百度开放平台进行分析，默认为启用（`True`）。如不使用可以修改为`False`。由于百度的并发限制较严格（QPS为2），使用百度进行分析时会较慢，约0.6s/条

**请注意**

需要分析的文本内容默认放置在一个`utf-8`编码的`CSV`**文件**的**第一列**内(不包含标题)。您也可以修改该函数的`colnum`
参数来指定获取数据的列。否则程序会报错。

通常情况下，该阿里云API的访问限制为50万次/天，并QPS为20。百度的API访问限制为50万次/年，QPS为2

### SentimentAnalysis函数

`SentimentAnalysis()`
函数可以用于分析单条文本并获得结果，用于处理批量分析时因网络或其他问题获得的错误结果。通常情况下，如果未能获得正确的结果，将在结果文件的第二列中呈现报错原因或代码。通常情况下，请检查网络连接或文本是否超出了1000个字。如果手动复制粘贴内容仍然获得报错代码，则可以访问[阿里云的这个页面](https://help.aliyun.com/document_detail/179345.html)或[百度的这个页面](https://ai.baidu.com/ai-doc/NLP/tk6z52b9z#%E9%94%99%E8%AF%AF%E7%A0%81)
查看错误码部分。或者联系*组会群中背景为白色，主体为一直坐着的大象的同学*。

## 解读结果

我们关注的内容主要集中在下面：

 - `text`：被分析的文本内容
 - `AliSentiment`： **阿里**分析判断的情感类型，包含正面、负面、中性三种情况
 - `AliPositive_prob`：句子极性为正面的概率大小，0-1之间，保留4位小数
 - `AliNeutral_prob`：句子极性为中性的概率大小，0-1之间，保留4位小数
 - `AliNegative_prob`：句子极性为负面的概率大小，0-1之间，保留4位小数
 - `AliRequestId`：阿里唯一请求id，排查问题的依据
 - `BaiduSentiment`：**百度**分析判断的情感类型，包含正向、负向、中性三种情况
 - `BaiduConfidence`：表示情感分类的置信度
 - `BaiduPositive_prob`：表示情感属于积极类别的概率
 - `BaiduNegative_prob`：表示情感属于消极类别的概率
 - `BaiduLogid`：百度记录id，排查问题的依据

## 引用

您可以考虑下面的引用格式

## 使用阿里云分析服务

阿里云. (2024, January 24). 自然语言处理NLP. https://ai.aliyun.com/nlp

或

aliyun. (2024, January 24). Natural language processing. https://ai.aliyun.com/nlp

## 使用百度开放平台分析服务

百度AI开放平台. (2024, January 28). 情感倾向分析. https://ai.baidu.com/tech/nlp_apply/sentiment_classify

或


Baidu AI Open Platform. (2024, January 28). Sentiment Analysis. https://ai.baidu.com/tech/nlp_apply/sentiment_classify

## 如果您想引用本项目

Xu, W. (2024). SentimentAnalysisWithAPI. (2024). GitHub repository. https://github.com/azanghai/SentimentAnalysisWithAPI/





