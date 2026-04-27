import json
import os
import csv
import time
import requests

from tqdm import tqdm
from aliyunsdkalinlp.request.v20200629 import GetSaChGeneralRequest
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.acs_exception.exceptions import ClientException
from aliyunsdkcore.acs_exception.exceptions import ServerException

from aip import AipNlp

import emoji

# ==================== 加载配置文件 ====================
try:
    from config import (
        ALI_ACCESS_KEY_ID, ALI_ACCESS_KEY_SECRET, ALI_REGION,
        BAIDU_APP_ID, BAIDU_API_KEY, BAIDU_SECRET_KEY,
        AIHUBMIX_API_KEY, AIHUBMIX_BASE_URL
    )
except ImportError:
    print("=" * 60)
    print("错误：未找到 config.py 配置文件！")
    print("请按以下步骤操作：")
    print("在 config.py 中填入你的真实密钥")
    print("=" * 60)
    raise SystemExit(1)

# ==================== 阿里云客户端 ====================
clientAli = AcsClient(
    ALI_ACCESS_KEY_ID,
    ALI_ACCESS_KEY_SECRET,
    ALI_REGION
)

# ==================== 百度客户端 ====================
clientBaidu = AipNlp(BAIDU_APP_ID, BAIDU_API_KEY, BAIDU_SECRET_KEY)

# ==================== 默认情感分析 Prompt ====================
DEFAULT_SENTIMENT_PROMPT = """你是一个情感分析专家。请对以下文本进行情感分析。
你只能从"积极"、"消极"、"中性"三个选项中选择一个作为结果。
请严格以JSON格式返回，不要包含任何其他文字、解释或markdown标记。
返回格式：
{{"sentiment": "积极/消极/中性"}}
待分析文本：{text}"""


# ==================== Emoji 统一预处理 ====================
def _process_emoji(text, emojitreat):
    if emojitreat == 'replace':
        return emoji.demojize(text, language='zh', delimiters=('', ''))
    elif emojitreat == 'delete':
        return emoji.replace_emoji(string=text, replace='')
    else:
        print('可能的emoji处理方式拼写错误，保持原文本')
        return text


# ==================== 阿里云情感分析 ====================
def SentimentAnalysisAli(text, emojitreat):
    request = GetSaChGeneralRequest.GetSaChGeneralRequest()
    text = _process_emoji(text, emojitreat)
    request.set_Text(text)
    request.set_ServiceCode("alinlp")
    response = clientAli.do_action_with_exception(request)
    resp_obj = json.loads(response)
    return resp_obj


# ==================== 百度情感分析 ====================
def SentimentAnalysisBaidu(text, emojitreat):
    text = _process_emoji(text, emojitreat)
    processed_text = text.lstrip('\ufeff').replace(':', '')
    gbk_encoded_text = processed_text.encode('utf-8').decode('utf-8')
    resp_obj = clientBaidu.sentimentClassify(gbk_encoded_text)
    return resp_obj


# ==================== GenAI 情感分析 ====================
def SentimentAnalysisGenAI(text, emojitreat,
                           model="gpt-4o-mini",
                           prompt_template=None,
                           api_key=None,
                           base_url=None,
                           max_retries=3):
    text = _process_emoji(text, emojitreat)

    if prompt_template is None:
        prompt_template = DEFAULT_SENTIMENT_PROMPT
    prompt = prompt_template.format(text=text)

    _api_key = api_key or AIHUBMIX_API_KEY
    _base_url = base_url or AIHUBMIX_BASE_URL

    headers = {
        "Authorization": f"Bearer {_api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(
                url=_base_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=60
            )
            response.raise_for_status()

            resp_json = response.json()
            content = resp_json['choices'][0]['message']['content'].strip()

            if content.startswith("```"):
                lines = content.split('\n')
                lines = [l for l in lines if not l.strip().startswith("```")]
                content = '\n'.join(lines).strip()

            try:
                result = json.loads(content)
                sentiment = result.get("sentiment", "").strip()
            except json.JSONDecodeError:
                sentiment = ""
                for keyword in ["积极", "消极", "中性"]:
                    if keyword in content:
                        sentiment = keyword
                        break

            if sentiment in ("积极", "消极", "中性"):
                return {
                    "success": True,
                    "sentiment": sentiment,
                    "model": model,
                    "raw_response": content
                }
            else:
                return {
                    "success": False,
                    "error": f"模型返回了非预期值: {sentiment}",
                    "raw_response": content,
                    "model": model
                }

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"[{model}] 请求失败，{wait_time}秒后重试 ({attempt + 1}/{max_retries}): {e}")
                time.sleep(wait_time)
            else:
                return {
                    "success": False,
                    "error": f"请求失败: {str(e)}",
                    "model": model
                }
        except (KeyError, IndexError) as e:
            return {
                "success": False,
                "error": f"响应解析失败: {str(e)}",
                "model": model
            }

    return {"success": False, "error": "未知错误", "model": model}


# ==================== 结果提取辅助函数 ====================
def _extract_ali_result(resultAli):
    headers = ['AliSentiment', 'AliPositive_prob', 'AliNeutral_prob', 'AliNegative_prob', 'AliRequestId']
    if 'Data' in resultAli:
        dataAli = json.loads(resultAli['Data'])['result']
        values = [dataAli['sentiment'], dataAli['positive_prob'], dataAli['neutral_prob'],
                  dataAli['negative_prob'], resultAli['RequestId']]
    else:
        values = [json.dumps(resultAli), None, None, None, None]
    return headers, values


def _extract_baidu_result(resultBaidu):
    headers = ['BaiduSentiment', 'BaiduConfidence', 'BaiduPositive_prob', 'BaiduNegative_prob', 'BaiduLogid']
    if 'items' in resultBaidu:
        dataBaidu = resultBaidu['items'][0]
        sentiment_map = {0: '消极', 1: '中性', 2: '积极'}
        dataBaidu['sentiment'] = sentiment_map.get(dataBaidu['sentiment'], dataBaidu['sentiment'])
        values = [dataBaidu['sentiment'], dataBaidu['confidence'], dataBaidu['positive_prob'],
                  dataBaidu['negative_prob'], resultBaidu['log_id']]
    else:
        values = [json.dumps(resultBaidu), None, None, None, None]
    return headers, values


def _extract_genai_result(resultGenAI, label):
    """
    提取 GenAI 结果。

    参数:
        resultGenAI: GenAI 返回的 dict
        label:       列名前缀，如 'GenAI_1', 'GenAI_2'
    """
    headers = [f'{label}_Sentiment', f'{label}_Model']
    if resultGenAI.get("success"):
        values = [resultGenAI['sentiment'], resultGenAI['model']]
    else:
        error_info = resultGenAI.get('error', '') + ' | ' + resultGenAI.get('raw_response', '')
        values = [error_info, resultGenAI.get('model', '')]
    return headers, values


# ==================== 模型标签生成工具 ====================
def _make_model_label(model_name, index):
    """
    根据模型名称生成简短列名前缀。

    示例:
        'gpt-4o-mini'         → 'GenAI_gpt4omini'
        'claude-sonnet-4-20250514' → 'GenAI_claudesonnet420250514'

    如果多个模型简化后冲突，则加上序号后缀。
    """
    short = model_name.replace('-', '').replace('.', '').replace('/', '_')
    return f"GenAI_{short}"


def _build_genai_labels(models_list):
    """
    为模型列表生成不重复的标签。
    返回: [(model_name, label), ...]
    """
    raw_labels = [_make_model_label(m, i) for i, m in enumerate(models_list)]

    # 检测冲突：如果有重复标签则加 _1, _2 后缀
    from collections import Counter
    counts = Counter(raw_labels)
    seen = {}
    final = []
    for model, label in zip(models_list, raw_labels):
        if counts[label] > 1:
            idx = seen.get(label, 0) + 1
            seen[label] = idx
            final.append((model, f"{label}_{idx}"))
        else:
            final.append((model, label))
    return final


# ==================== 解析 genai_prompt 参数 ====================
def _resolve_prompt(genai_prompt, model_name):
    """
    根据 genai_prompt 的类型，解析出当前模型应使用的 prompt。

    genai_prompt 支持三种写法:
        None           → 使用默认 prompt
        str            → 所有模型共用这一个 prompt
        dict           → 按模型名映射，未命中则用默认
    """
    if genai_prompt is None:
        return None  # SentimentAnalysisGenAI 内部会 fallback 到默认
    elif isinstance(genai_prompt, str):
        return genai_prompt
    elif isinstance(genai_prompt, dict):
        return genai_prompt.get(model_name, None)
    else:
        return None


# ==================== 主分析流程 ====================
def StartAnalysis(input_file, output_file, colnum=1,
                  Ali=True, Baidu=True, GenAI=False,
                  emojitreat='replace',
                  has_header=True,
                  # ---- GenAI 参数（支持多模型）----
                  genai_models='gpt-4o-mini',
                  genai_prompt=None,
                  genai_api_key=None,
                  genai_base_url=None,
                  genai_sleep=0.5):
    """
    整体情感分析处理，在原表所有列的基础上追加分析结果列。

    参数:
        input_file:     输入 CSV 文件路径
        output_file:    输出 CSV 文件路径
        colnum:         文本所在列号 (从1开始)
        Ali:            是否使用阿里云
        Baidu:          是否使用百度
        GenAI:          是否使用 GenAI
        emojitreat:     emoji 处理方式 ('replace' / 'delete')
        has_header:     输入文件是否包含表头行 (True/False)

        genai_models:   GenAI 模型，支持三种写法:
                          - 单个字符串:  'gpt-4o-mini'
                          - 多个模型列表: ['gpt-4o-mini', 'claude-sonnet-4-20250514', 'deepseek-chat']
                        （旧参数名 genai_model 仍兼容，见下方）

        genai_prompt:   GenAI prompt，支持三种写法:
                          - None:  所有模型使用默认 prompt
                          - str:   所有模型共用同一个自定义 prompt
                          - dict:  按模型名指定不同 prompt
                                   {'gpt-4o-mini': prompt_a, 'deepseek-chat': prompt_b}
                                   未指定的模型使用默认 prompt

        genai_api_key:  API Key，默认使用 config.py 中的 AIHUBMIX_API_KEY
        genai_base_url: API 地址，默认使用 config.py 中的 AIHUBMIX_BASE_URL
        genai_sleep:    每次 GenAI 请求后的等待时间(秒)，防止速率限制
    """
    colnum = int(colnum)

    if not (Ali or Baidu or GenAI):
        print('参数输入错误，请至少选择一个分析平台（阿里、百度或GenAI）')
        return

    # ---- 标准化 genai_models 为列表 ----
    if isinstance(genai_models, str):
        models_list = [genai_models]
    elif isinstance(genai_models, (list, tuple)):
        models_list = list(genai_models)
    else:
        models_list = [str(genai_models)]

    # 为每个模型生成列名标签
    model_labels = _build_genai_labels(models_list)  # [(model_name, label), ...]

    # ---- 构建需要追加的新列表头 ----
    new_headers = []
    if Ali:
        new_headers.extend(['AliSentiment', 'AliPositive_prob', 'AliNeutral_prob',
                            'AliNegative_prob', 'AliRequestId'])
    if Baidu:
        new_headers.extend(['BaiduSentiment', 'BaiduConfidence', 'BaiduPositive_prob',
                            'BaiduNegative_prob', 'BaiduLogid'])
    if GenAI:
        for model_name, label in model_labels:
            new_headers.extend([f'{label}_Sentiment', f'{label}_Model'])

    # ---- 打印分析配置 ----
    print("=" * 60)
    print("情感分析配置:")
    print(f"  输入文件: {input_file}")
    print(f"  输出文件: {output_file}")
    print(f"  文本列号: {colnum}")
    print(f"  阿里云: {'✓' if Ali else '✗'}")
    print(f"  百度:   {'✓' if Baidu else '✗'}")
    if GenAI:
        print(f"  GenAI:  ✓ ({len(models_list)} 个模型)")
        for model_name, label in model_labels:
            prompt_type = "自定义" if _resolve_prompt(genai_prompt, model_name) else "默认"
            print(f"    - {model_name} (列前缀: {label}, prompt: {prompt_type})")
    else:
        print(f"  GenAI:  ✗")
    print("=" * 60)

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # ---------- 处理表头 ----------
        if has_header:
            original_header = next(reader)
            writer.writerow(original_header + new_headers)
        else:
            first_row = next(reader)
            placeholder_header = [f'col_{i + 1}' for i in range(len(first_row))]
            writer.writerow(placeholder_header + new_headers)
            infile.seek(0)
            reader = csv.reader(infile)

        # ---------- 逐行处理 ----------
        for row in tqdm(reader, desc="Processing"):
            updated_row = list(row)

            try:
                text = row[colnum - 1]
            except IndexError:
                print(f"警告：第 {colnum} 列不存在，跳过此行: {row}")
                updated_row.extend([None] * len(new_headers))
                writer.writerow(updated_row)
                continue

            if not text.strip():
                updated_row.extend([None] * len(new_headers))
                writer.writerow(updated_row)
                continue

            # 阿里云分析
            if Ali:
                resultAli = SentimentAnalysisAli(text, emojitreat)
                _, ali_values = _extract_ali_result(resultAli)
                updated_row.extend(ali_values)

            # 百度分析
            if Baidu:
                resultBaidu = SentimentAnalysisBaidu(text, emojitreat)
                _, baidu_values = _extract_baidu_result(resultBaidu)
                updated_row.extend(baidu_values)
                time.sleep(0.6)

            # GenAI 分析 —— 逐个模型调用
            if GenAI:
                for model_name, label in model_labels:
                    current_prompt = _resolve_prompt(genai_prompt, model_name)
                    resultGenAI = SentimentAnalysisGenAI(
                        text=text,
                        emojitreat=emojitreat,
                        model=model_name,
                        prompt_template=current_prompt,
                        api_key=genai_api_key,
                        base_url=genai_base_url
                    )
                    _, genai_values = _extract_genai_result(resultGenAI, label)
                    updated_row.extend(genai_values)
                    time.sleep(genai_sleep)

            writer.writerow(updated_row)

    print(f"\n分析完成！结果已保存到: {output_file}")


if __name__ == "__main__":

    # ========== 示例1: 同时使用 3 个 GenAI 模型 + 阿里 + 百度 ==========
    prompt_1 = """请判断下面文本的情感倾向。
    只能回答"积极""消极""中性"之一，以JSON返回：{{"sentiment":"你的判断"}}
    文本：{text}"""
    prompt_2 = """你是一个情感分析专家。请对以下文本进行情感分析。
    你只能从"积极"、"消极"、"中性"三个选项中选择一个作为结果。
    请严格以JSON格式返回，不要包含任何其他文字、解释或markdown标记。
    返回格式：
    {{"sentiment": "积极/消极/中性"}}
    待分析文本：{text}
    """
    prompt_3 = """
    请判断下面文本的情感倾向。并给出相反的答案
    只能回答"积极""消极""中性"之一，以JSON返回：{{"sentiment":"你的判断"}}
    文本：{text}
    """

    StartAnalysis(
        input_file='../TestFiles/TestFIle.csv',
        output_file='../Results/TestResult_multi.csv',
        colnum=3,
        Ali=True,
        Baidu=True,
        GenAI=True,
        emojitreat='replace',
        has_header=True,
        genai_models=['gpt-4o-mini', 'gpt-5.4-mini', 'deepseek-v4-flash'],
        genai_prompt={'gpt-40-mini':prompt_1, 'gpt-5.4-mini':prompt_2, 'deepseek-v4-flash':prompt_3},
        genai_sleep=0.2
    )
    # 输出列: ...原表列... | Ali列 | Baidu列 | GenAI_gpt4omini_Sentiment | GenAI_gpt4omini_Model | GenAI_claudesonnet420250514_Sentiment | GenAI_claudesonnet420250514_Model | GenAI_deepseekchat_Sentiment | GenAI_deepseekchat_Model

    # ========== 示例2: 两个模型，各用不同 prompt ==========
    # prompt_gpt = """请判断下面文本的情感倾向。
    # 只能回答"积极""消极""中性"之一，以JSON返回：{{"sentiment":"你的判断"}}
    # 文本：{text}"""
    #
    # prompt_claude = """Analyze the sentiment of the following Chinese text.
    # Reply ONLY with JSON: {{"sentiment": "积极/消极/中性"}}
    # Text: {text}"""
    #
    # StartAnalysis(
    #     input_file='../TestFiles/TestFIle.csv',
    #     output_file='../Results/TestResult_diff_prompt.csv',
    #     colnum=3,
    #     Ali=False,
    #     Baidu=False,
    #     GenAI=True,
    #     emojitreat='replace',
    #     has_header=True,
    #     genai_models=['gpt-4o-mini', 'claude-sonnet-4-20250514'],
    #     genai_prompt={
    #         'gpt-4o-mini': prompt_gpt,
    #         'claude-sonnet-4-20250514': prompt_claude
    #     },
    #     genai_sleep=0.5
    # )

    # ========== 示例3: 单个模型（向后兼容，和以前写法一样） ==========
    # StartAnalysis(
    #     input_file='../TestFiles/TestFIle.csv',
    #     output_file='../Results/TestResult_single.csv',
    #     colnum=3,
    #     Ali=True,
    #     Baidu=True,
    #     GenAI=True,
    #     emojitreat='replace',
    #     has_header=True,
    #     genai_models='gpt-4o-mini'
    # )

    # ========== 示例4: 单条测试多模型 ==========
    # for m in ['gpt-4o-mini', 'deepseek-chat']:
    #     result = SentimentAnalysisGenAI(
    #         text='这家餐厅的服务态度真好，菜品也很美味！',
    #         emojitreat='replace',
    #         model=m
    #     )
    #     print(f"[{m}] {result}")