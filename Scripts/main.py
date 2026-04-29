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
    try:                                                            # <== 新增 try
        if 'Data' in resultAli:
            data_parsed = json.loads(resultAli['Data'])
            if 'result' in data_parsed:                             # <== 新增检查
                dataAli = data_parsed['result']
                values = [dataAli['sentiment'], dataAli['positive_prob'], dataAli['neutral_prob'],
                          dataAli['negative_prob'], resultAli.get('RequestId')]
            else:
                # Data 存在但无 result（限流/异常响应等）              <== 新增分支
                values = [f"AliError: {json.dumps(data_parsed, ensure_ascii=False)}",
                          None, None, None, resultAli.get('RequestId')]
        else:
            values = [f"AliError: {json.dumps(resultAli, ensure_ascii=False)}",
                      None, None, None, None]
    except Exception as e:                                          # <== 新增兜底
        values = [f"AliException: {str(e)}", None, None, None, None]
    return headers, values
def _extract_baidu_result(resultBaidu):
    headers = ['BaiduSentiment', 'BaiduConfidence', 'BaiduPositive_prob', 'BaiduNegative_prob', 'BaiduLogid']
    try:                                                            # <== 新增 try
        if 'items' in resultBaidu and len(resultBaidu['items']) > 0:  # <== 加 len 检查
            dataBaidu = resultBaidu['items'][0]
            sentiment_map = {0: '消极', 1: '中性', 2: '积极'}
            sentiment_val = sentiment_map.get(dataBaidu.get('sentiment'), dataBaidu.get('sentiment'))
            values = [sentiment_val, dataBaidu.get('confidence'), dataBaidu.get('positive_prob'),
                      dataBaidu.get('negative_prob'), resultBaidu.get('log_id')]
        else:
            values = [f"BaiduError: {json.dumps(resultBaidu, ensure_ascii=False)}",
                      None, None, None, resultBaidu.get('log_id')]
    except Exception as e:                                          # <== 新增兜底
        values = [f"BaiduException: {str(e)}", None, None, None, None]
    return headers, values
def _extract_genai_result(resultGenAI, label):
    headers = [f'{label}_Sentiment', f'{label}_Model']
    try:                                                            # <== 新增 try
        if resultGenAI.get("success"):
            values = [resultGenAI['sentiment'], resultGenAI['model']]
        else:
            error_info = resultGenAI.get('error', '') + ' | ' + resultGenAI.get('raw_response', '')
            values = [error_info, resultGenAI.get('model', '')]
    except Exception as e:                                          # <== 新增兜底
        values = [f"GenAIException: {str(e)}", '']
    return headers, values


# ==================== 模型标签生成工具 ====================
def _make_model_label(model_name, index):
    short = model_name.replace('-', '').replace('.', '').replace('/', '_')
    return f"GenAI_{short}"


def _build_genai_labels(models_list):
    raw_labels = [_make_model_label(m, i) for i, m in enumerate(models_list)]
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
    if genai_prompt is None:
        return None
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
                  resume=True,                                      # <== 新增参数
                  genai_models='gpt-4o-mini',
                  genai_prompt=None,
                  genai_api_key=None,
                  genai_base_url=None,
                  genai_sleep=0.5):
    """
        整体情感分析处理，在原表所有列的基础上追加分析结果列。
        每处理完一行立即写入并刷盘；支持断点续传，中途中断后重新运行即可自动接续。
        新增参数:
        resume:  是否启用断点续传 (默认 True)
                 - True:  若输出文件已存在，自动跳过已完成行，追加写入
                 - False: 忽略已有输出文件，从头覆盖重新分析
                 注意：续传要求分析配置（平台选择、模型列表等）与上次一致，
                       若配置变更导致列数不匹配，会自动回退到从头开始。
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

    model_labels = _build_genai_labels(models_list)

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

    # ======================================================================
    #  计算输入文件总数据行数（用 csv.reader 正确处理含换行的字段）        <== 新增
    # ======================================================================
    with open(input_file, 'r', encoding='utf-8') as f:
        total_input_lines = sum(1 for _ in csv.reader(f))
    total_data_rows = total_input_lines - (1 if has_header else 0)

    # ======================================================================
    #  断点续传检测                                                        <== 新增
    # ======================================================================
    skip_rows = 0
    file_mode = 'w'          # 默认：覆盖写
    write_header = True       # 默认：需要写表头

    if resume and os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                out_reader = csv.reader(f)
                existing_header = next(out_reader)          # 读已有表头
                completed_rows = sum(1 for _ in out_reader)  # 已完成数据行

            # — 验证列数是否与当前配置匹配 —
            with open(input_file, 'r', encoding='utf-8') as f:
                first_input_row = next(csv.reader(f))
            expected_total_cols = len(first_input_row) + len(new_headers)

            if len(existing_header) != expected_total_cols:
                print(f"  ⚠ 已有输出文件列数({len(existing_header)})"
                      f"与当前配置期望({expected_total_cols})不匹配，配置可能已变更")
                print(f"    将覆盖重新开始分析")
                # 保持默认 file_mode='w', write_header=True, skip_rows=0

            elif completed_rows >= total_data_rows:
                print("=" * 60)
                print(f"  ✓ 所有 {total_data_rows} 行已分析完毕，无需继续。")
                print(f"    如需重新分析，请删除输出文件或设置 resume=False")
                print("=" * 60)
                return

            elif completed_rows > 0:
                skip_rows = completed_rows
                file_mode = 'a'       # 追加模式
                write_header = False   # 表头已存在

            else:
                # 输出文件仅有表头，无数据行
                file_mode = 'a'
                write_header = False

        except Exception as e:
            print(f"  ⚠ 读取已有输出文件失败({e})，将覆盖重新开始")
            # 保持默认值

    remaining_rows = total_data_rows - skip_rows                    # <== 新增

    # ---- 打印分析配置 ----
    print("=" * 60)
    print("情感分析配置:")
    print(f"  输入文件: {input_file}")
    print(f"  输出文件: {output_file}")
    print(f"  文本列号: {colnum}")
    print(f"  总数据行: {total_data_rows}")                          # <== 新增
    print(f"  阿里云: {'✓' if Ali else '✗'}")
    print(f"  百度:   {'✓' if Baidu else '✗'}")
    if GenAI:
        print(f"  GenAI:  ✓ ({len(models_list)} 个模型)")
        for model_name, label in model_labels:
            prompt_type = "自定义" if _resolve_prompt(genai_prompt, model_name) else "默认"
            print(f"    - {model_name} (列前缀: {label}, prompt: {prompt_type})")
    else:
        print(f"  GenAI:  ✗")
    print(f"  逐行即时保存: ✓")
    # ---- 断点续传状态 ----                                          <== 新增
    if skip_rows > 0:
        print(f"  断点续传: ✓ 已完成 {skip_rows}/{total_data_rows} 行，"
              f"本次继续剩余 {remaining_rows} 行")
    elif resume and file_mode == 'a':
        print(f"  断点续传: ✓ 表头已存在，从第 1 行开始 (共 {total_data_rows} 行)")
    else:
        status = "已启用（未发现已有进度）" if resume else "未启用"
        print(f"  断点续传: {status}")
    print("=" * 60)

    # ==================================================================
    #  主处理循环
    # ==================================================================
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, file_mode, newline='', encoding='utf-8') as outfile:  # <== file_mode

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # ---------- 处理表头 ----------
        if has_header:
            original_header = next(reader)              # 消耗输入文件表头行
            if write_header:                            # <== 新增条件
                writer.writerow(original_header + new_headers)
                outfile.flush()
        else:
            if write_header:                            # <== 新增条件
                first_row = next(reader)
                placeholder_header = [f'col_{i + 1}' for i in range(len(first_row))]
                writer.writerow(placeholder_header + new_headers)
                outfile.flush()
                infile.seek(0)
                reader = csv.reader(infile)
            # 若 write_header=False (续传)，reader 从文件头开始，所有行都是数据

        # ---------- 跳过已完成的行 ----------                        <== 新增
        for i in range(skip_rows):
            try:
                next(reader)
            except StopIteration:
                print(f"警告：输入文件行数不足，跳过 {i} 行后已到末尾，无需继续")
                return

        if skip_rows > 0:
            print(f"已跳过前 {skip_rows} 行已完成数据，开始继续分析...")

        # ---------- 逐行处理剩余数据 ----------
        pbar = tqdm(total=total_data_rows, initial=skip_rows,
                    desc="Processing")

        for row in reader:
            updated_row = list(row)

            try:
                text = row[colnum - 1]
            except IndexError:
                print(f"警告：第 {colnum} 列不存在，跳过此行: {row}")
                updated_row.extend([None] * len(new_headers))
                writer.writerow(updated_row)
                outfile.flush()
                pbar.update(1)
                continue

            if not text.strip():
                updated_row.extend([None] * len(new_headers))
                writer.writerow(updated_row)
                outfile.flush()
                pbar.update(1)
                continue

            # ============================================================
            #  阿里云分析（带异常保护）                                   <== 修改
            # ============================================================
            if Ali:
                try:
                    resultAli = SentimentAnalysisAli(text, emojitreat)
                except Exception as e:
                    resultAli = {'AliCallError': str(e)}
                    print(f"\n  ⚠ 阿里云调用异常: {e}")
                _, ali_values = _extract_ali_result(resultAli)
                updated_row.extend(ali_values)

            # ============================================================
            #  百度分析（带异常保护）                                     <== 修改
            # ============================================================
            if Baidu:
                try:
                    resultBaidu = SentimentAnalysisBaidu(text, emojitreat)
                except Exception as e:
                    resultBaidu = {'BaiduCallError': str(e)}
                    print(f"\n  ⚠ 百度调用异常: {e}")
                _, baidu_values = _extract_baidu_result(resultBaidu)
                updated_row.extend(baidu_values)
                time.sleep(0.6)

            # ============================================================
            #  GenAI 分析（带异常保护）                                   <== 修改
            # ============================================================
            if GenAI:
                for model_name, label in model_labels:
                    try:
                        current_prompt = _resolve_prompt(genai_prompt, model_name)
                        resultGenAI = SentimentAnalysisGenAI(
                            text=text,
                            emojitreat=emojitreat,
                            model=model_name,
                            prompt_template=current_prompt,
                            api_key=genai_api_key,
                            base_url=genai_base_url
                        )
                    except Exception as e:
                        resultGenAI = {'success': False, 'error': f'调用异常: {str(e)}',
                                       'model': model_name}
                        print(f"\n  ⚠ GenAI [{model_name}] 调用异常: {e}")
                    _, genai_values = _extract_genai_result(resultGenAI, label)
                    updated_row.extend(genai_values)
                    time.sleep(genai_sleep)

            writer.writerow(updated_row)
            outfile.flush()
            pbar.update(1)

        pbar.close()

    print(f"\n分析完成！结果已保存到: {output_file}")


if __name__ == "__main__":

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
        input_file=r'C:\Users\x5058\MOFANGSync\SemAppPsy\Paper\Data\combined_weibo_all_216.csv',
        output_file='../Results/Weibo216_multi.csv',
        colnum=3,
        Ali=True,
        Baidu=True,
        GenAI=True,
        emojitreat='replace',
        has_header=True,
        resume=True,                # <== 断点续传开关，默认开启
        genai_models=['gpt-5.4-nano', 'qwen3.6-flash', 'deepseek-v4-pro', 'claude-sonnet-4-6'],
        genai_prompt={'gpt-5.4-nano': prompt_2, 'qwen3.6-flash': prompt_2,
                      'deepseek-v4-pro': prompt_2, 'claude-sonnet-4-6': prompt_2},
        genai_sleep=0.2
    )