# 使用API服务进行情感分析

> 2026.4.28更新：新增 GenAI（大语言模型）多模型情感分析支持；重构密钥管理与整体架构  
> 2024.2.1更新：增加了处理emoji表情的能力；百度平台改用UTF-8编码  
> 2024.1.28更新：增加了百度开放平台的识别，准确率更高


目前，该脚本支持以下三类情感分析：

| 平台                                                              | 说明 |
|-----------------------------------------------------------------|------|
| [阿里云 NLP](https://help.aliyun.com/document_detail/179345.html)  | 传统 NLP 情感分析 |
| [百度开放平台](https://ai.baidu.com/tech/nlp_apply/sentiment_classify) | 传统 NLP 情感分析 |
| **GenAI（新增，默认通过[aihubmix](https://aihubmix.com)）** | 通过大语言模型 API（GPT、Claude、DeepSeek 等）进行情感分析，支持**同时调用多个模型**并自定义 Prompt |

相较于自行撰写代码调用 `API`，该脚本提供了一个开箱即用（_仅需简单配置_）的批量情感分析工具。

## 在开始之前

首先，您需要根据想使用的服务获取对应的密钥：

**阿里云**：`Access Key ID` 和 `Access Key Secret`  
**百度**：`APP_ID`、`API_KEY` 和 `SECRET_KEY`  
**GenAI**：兼容 `OpenAI` 接口的 `API Key` 和 `Base URL`（如 `AiHubMix`、`OpenAI` 官方等）  
**好消息是** 如果您是 _杨老师课题组_ 的成员或 _杨老师_ 的学生，请您联络组会群中头像为一只大象在白色背景中的成员，或者直接联络杨老师获取配置的密钥。

该脚本的撰写和测试使用 `Python 3.10` 版本。

### 安装运行脚本所需要的库

您需要正确安装运行脚本所需的包。如果您使用的是PyCharm，则您在加载本项目时应当会有提示进行自动安装，否则，您需要手动安装，您可以在终端中分别运行下面的代码来安装：

```
pip install aliyun-python-sdk-alinlp
pip install tqdm
pip install baidu-aip
pip install requests
pip install emoji
```

在运行之前，请找到脚本（Scripts文件夹下的main.py文件），并修改下面的内容：

### 配置密钥（config.py）

本项目使用独立的 `config.py` 文件管理所有密钥，不再需要在 `main.py` 中直接修改代码。

在 `Scripts` 文件夹下找到 `config_template.py`（或自行创建 `config.py`）；
将其复制/重命名为 `config.py`；
在 `config.py` 中填入你的真实密钥：

```
# ==================== 阿里云 ====================
ALI_ACCESS_KEY_ID = '你的阿里云 Access Key ID'
ALI_ACCESS_KEY_SECRET = '你的阿里云 Access Key Secret'
ALI_REGION = 'cn-hangzhou'

# ==================== 百度 ====================
BAIDU_APP_ID = '你的百度 APP_ID'
BAIDU_API_KEY = '你的百度 API_KEY'
BAIDU_SECRET_KEY = '你的百度 SECRET_KEY'

# ==================== GenAI (兼容 OpenAI 接口) ====================
AIHUBMIX_API_KEY = '你的 API Key'
AIHUBMIX_BASE_URL = 'https://aihubmix.com/v1/chat/completions'
```

> **请注意**： 请不要将 `config.py` 或其中的密钥分享给其他任何人或上传至网络。建议在 `.gitignore` 中添加 `config.py`。

## 开始使用

请打开 `Scripts/main.py`，划至脚本最后的 `if __name__ == "__main__":` 下方修改参数并运行。

### StartAnalysis函数

`StartAnalysis() `是主要的批量分析函数，参数说明如下：

 | 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `input_file` | str | — | 传入的 CSV 文件路径。默认分析文件为 `TestFiles/TestFile.csv`（包含测试文本） |
| `output_file` | str | — | 分析结果的输出路径。默认保存在 `Results/` 文件夹下。如不改名，每次运行会覆写 |
| `colnum` | int | `1` | 文本所在列号（从 1 开始计数） |
| `Ali` | bool | `True` | 是否使用阿里云进行分析 |
| `Baidu` | bool | `True` | 是否使用百度开放平台进行分析。由于百度 QPS 限制为 2，分析速度约 0.6s/条 |
| `GenAI` | bool | `False` | 是否使用 GenAI（大语言模型）进行分析 |
| `emojitreat` | str | `'replace'` | Emoji 处理方式：`'replace'` 将 emoji 映射为中文描述；`'delete'` 删除 emoji |
| `has_header` | bool | `True` | 输入 CSV 是否包含表头行。如果为 `True`，第一行会被视为表头而非数据 |

### GenAI 专用参数

| 参数 | 类型 | 默认值 | 说明                                                                                                   |
|------|------|--------|------------------------------------------------------------------------------------------------------|
| `genai_models` | str 或 list | `'gpt-4o-mini'` | 要使用的模型。支持单个字符串或模型列表，如 `['gpt-4o-mini', 'deepseek-chat']`,更多模型可以点击[这里](https://aihubmix.com/models)查看 |
| `genai_prompt` | None / str / dict | `None` | 自定义 Prompt（详见下方说明）                                                                                   |
| `genai_api_key` | str | `None` | 覆盖 `config.py` 中的 `API Key`                                                                            |
| `genai_base_url` | str | `None` | 覆盖 `config.py` 中的 API 地址                                                                             |
| `genai_sleep` | float | `0.5` | 每次 GenAI 请求后的等待时间（秒），用于防止速率限制                                                                        |

### genai_prompt 的三种写法

| 写法 | 效果 |
|------|------|
| `None`（默认） | 所有模型使用内置默认 Prompt |
| 字符串 | 所有模型共用同一个自定义 Prompt |
| 字典 | 按模型名指定不同 Prompt，未指定的模型使用默认 Prompt |

自定义 Prompt 中使用 {text} 作为待分析文本的占位符。示例：

```python
# 所有模型共用
genai_prompt = """请判断下面文本的情感倾向。
只能回答"积极""消极""中性"之一，以JSON返回：{{"sentiment":"你的判断"}}
文本：{text}"""

# 按模型指定
genai_prompt = {
    'gpt-4o-mini': prompt_gpt,
    'deepseek-chat': prompt_deepseek
}
```

> **注意**: 
> 1. Prompt 中的花括号如果是 JSON 示例的一部分，需要使用双花括号 {{ }} 进行转义，只有 {text} 会被替换为实际文本。
> 2. 模型的名称必须与 `genai_models` 中指定的完全匹配，区分大小写。具体模型名称可以前往 [AiHubMix 模型列表](https://aihubmix.com/models) 查看。
> 3. 在先前的报道中发现部分API中转站可能会提供虚假的模型（例如使用低价低性能模型替代高价高性能模型，token掺水等），本项目的默认API中转站经试用与检测后可信度较高。

> `aihubmix` 似乎支持来自项目调用的10%优惠，将计划在未来的更新中应用。

### 关于 Temperature 参数

GenAI 情感分析在调用大语言模型时，`temperature` 固定设置为 `0.0`。`Temperature` 控制模型输出的随机性：

 - `temperature = 0.0`：模型将（近乎）确定性地选择概率最高的输出，最大程度保证结果的一致性和可复现性；  
 - 较高的 `temperature`（如 `0.7`、`1.0`）：输出更具随机性和多样性，不适合需要稳定判断的情感分析任务。  
 - 对于情感分析这类分类任务，我们需要的是稳定、可复现的判断结果，因此默认将 `temperature` 设为 0。如需修改，可在 `SentimentAnalysisGenAI()` 函数内调整 `payload` 中的 `temperature` 值。

## 使用示例

```python
# 示例 1：同时使用阿里 + 百度 + 3个GenAI模型
StartAnalysis(
    input_file='../TestFiles/TestFile.csv',
    output_file='../Results/TestResult_multi.csv',
    colnum=3,
    Ali=True,
    Baidu=True,
    GenAI=True,
    emojitreat='replace',
    has_header=True,
    genai_models=['gpt-4o-mini', 'claude-sonnet-4-20250514', 'deepseek-chat'],
    genai_prompt={
        'gpt-4o-mini': prompt_gpt,
        'deepseek-chat': prompt_deepseek
        # claude 未指定，将使用默认 prompt
    },
    genai_sleep=0.5
)

# 示例 2：仅使用单个GenAI模型（向后兼容写法）
StartAnalysis(
    input_file='../TestFiles/TestFile.csv',
    output_file='../Results/TestResult_single.csv',
    colnum=3,
    Ali=True,
    Baidu=True,
    GenAI=True,
    emojitreat='replace',
    has_header=True,
    genai_models='gpt-4o-mini'
)

# 示例 3：仅使用阿里 + 百度（与旧版兼容）
StartAnalysis(
    input_file='../TestFiles/TestFile.csv',
    output_file='../Results/TestResult.csv',
    colnum=1,
    Ali=True,
    Baidu=True,
    GenAI=False,
    emojitreat='replace'
)
```
> **请注意：**
>  - 需要分析的文本内容应放置在 `UTF-8` 编码的 `CSV` 文件中；
> - 输出文件将保留原始 `CSV` 的所有列，并在末尾追加分析结果列；
> - 通常情况下，阿里云 API 的访问限制为 50 万次/天，QPS 为 20；百度 API 的访问为按此计费年，QPS 为 2；GenAI 的限制取决于所使用的具体服务商和模型。

### SentimentAnalysis 函数（单条分析）

可以直接调用以下函数对单条文本进行分析：

```python
# 阿里云
result = SentimentAnalysisAli(text='这是一个测试样例', emojitreat='replace')

# 百度
result = SentimentAnalysisBaidu(text='这是一个测试样例', emojitreat='replace')

# GenAI
result = SentimentAnalysisGenAI(
    text='这家餐厅的服务态度真好，菜品也很美味！',
    emojitreat='replace',
    model='gpt-4o-mini'
)
```

通常情况下，如果批量分析时未能获得正确的结果，将在结果文件对应列中呈现报错原因或代码。请检查网络连接或文本是否超出了字数限制。可访问阿里云错误码页面或百度错误码页面查看详情。或者联系*组会群中背景为白色，主体为一只坐着的大象的同学*。
您亦可在本项目中提出issues，或直接联系我。

## 解读结果

输出文件将保留输入 CSV 的所有原始列，并在末尾追加以下分析结果列：

### 阿里云分析结果列

| 列名 | 说明 |
|------|------|
| `AliSentiment` | 阿里分析判断的情感类型，包含正面、负面、中性 |
| `AliPositive_prob` | 句子极性为正面的概率，0-1 之间，保留 4 位小数 |
| `AliNeutral_prob` | 句子极性为中性的概率，0-1 之间，保留 4 位小数 |
| `AliNegative_prob` | 句子极性为负面的概率，0-1 之间，保留 4 位小数 |
| `AliRequestId` | 阿里唯一请求 ID，用于排查问题 |

### 百度分析结果列

| 列名 | 说明 |
|------|------|
| `BaiduSentiment` | 百度分析判断的情感类型，包含积极、消极、中性 |
| `BaiduConfidence` | 情感分类的置信度 |
| `BaiduPositive_prob` | 情感属于积极类别的概率 |
| `BaiduNegative_prob` | 情感属于消极类别的概率 |
| `BaiduLogid` | 百度记录 ID，用于排查问题 |

### GenAI 分析结果列

每个 GenAI 模型会生成两列，列名前缀根据模型名称自动生成（如 GenAI_gpt4omini）：

| 列名 | 说明 |
|------|------|
| `{label}_Sentiment` | 模型判断的情感类型：积极、消极、中性 |
| `{label}_Model` | 使用的模型名称 |

例如同时使用 `gpt-4o-mini` 和 `deepseek-chat` 时，输出列为：

|...原始列... | AliXxx | BaiduXxx | GenAI_gpt4omini_Sentiment | GenAI_gpt4omini_Model | GenAI_deepseekchat_Sentiment | GenAI_deepseekchat_Model |
|------|------|------|------|------|------|------|

## 引用

您可以考虑下面的引用格式

### 使用阿里云分析服务

阿里云. (2024, January 24). 自然语言处理NLP. https://ai.aliyun.com/nlp

或

aliyun. (2024, January 24). Natural language processing. https://ai.aliyun.com/nlp

### 使用百度开放平台分析服务

百度AI开放平台. (2024, January 28). 情感倾向分析. https://ai.baidu.com/tech/nlp_apply/sentiment_classify

或


Baidu AI Open Platform. (2024, January 28). Sentiment Analysis. https://ai.baidu.com/tech/nlp_apply/sentiment_classify

### 使用GenAI（大语言模型）分析服务

#### 如果您使用了 AiHubMix 的 GenAI 服务

AiHubMix. (2026). GenAI Models. https://aihubmix.com/models

### 如果您想引用本项目

Xu, W. (2024). SentimentAnalysisWithAPI. (2024). GitHub repository. https://github.com/azanghai/SentimentAnalysisWithAPI/





