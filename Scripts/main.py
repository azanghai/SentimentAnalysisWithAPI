import json
import os
import csv
import time
import shutil
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
    try:
        if 'Data' in resultAli:
            data_parsed = json.loads(resultAli['Data'])
            if 'result' in data_parsed:
                dataAli = data_parsed['result']
                values = [dataAli['sentiment'], dataAli['positive_prob'], dataAli['neutral_prob'],
                          dataAli['negative_prob'], resultAli.get('RequestId')]
            else:
                values = [f"AliError: {json.dumps(data_parsed, ensure_ascii=False)}",
                          None, None, None, resultAli.get('RequestId')]
        else:
            values = [f"AliError: {json.dumps(resultAli, ensure_ascii=False)}",
                      None, None, None, None]
    except Exception as e:
        values = [f"AliException: {str(e)}", None, None, None, None]
    return headers, values


def _extract_baidu_result(resultBaidu):
    headers = ['BaiduSentiment', 'BaiduConfidence', 'BaiduPositive_prob', 'BaiduNegative_prob', 'BaiduLogid']
    try:
        if 'items' in resultBaidu and len(resultBaidu['items']) > 0:
            dataBaidu = resultBaidu['items'][0]
            sentiment_map = {0: '消极', 1: '中性', 2: '积极'}
            sentiment_val = sentiment_map.get(dataBaidu.get('sentiment'), dataBaidu.get('sentiment'))
            values = [sentiment_val, dataBaidu.get('confidence'), dataBaidu.get('positive_prob'),
                      dataBaidu.get('negative_prob'), resultBaidu.get('log_id')]
        else:
            values = [f"BaiduError: {json.dumps(resultBaidu, ensure_ascii=False)}",
                      None, None, None, resultBaidu.get('log_id')]
    except Exception as e:
        values = [f"BaiduException: {str(e)}", None, None, None, None]
    return headers, values


def _extract_genai_result(resultGenAI, label):
    headers = [f'{label}_Sentiment', f'{label}_Model']
    try:
        if resultGenAI.get("success"):
            values = [resultGenAI['sentiment'], resultGenAI['model']]
        else:
            error_info = resultGenAI.get('error', '') + ' | ' + resultGenAI.get('raw_response', '')
            values = [error_info, resultGenAI.get('model', '')]
    except Exception as e:
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


# ==================== 平台结果有效性检查 (新增) ====================
def _is_valid_ali(row, col_start, col_count=5):
    """检查该行中阿里云结果是否有效（非空、非错误）"""
    if col_start is None or col_start + col_count > len(row):
        return False
    val = str(row[col_start]).strip() if row[col_start] else ''
    if not val:
        return False
    if 'Error' in val or 'Exception' in val:
        return False
    return True


def _is_valid_baidu(row, col_start, col_count=5):
    """检查该行中百度结果是否有效（非空、非错误）"""
    if col_start is None or col_start + col_count > len(row):
        return False
    val = str(row[col_start]).strip() if row[col_start] else ''
    if not val:
        return False
    if 'Error' in val or 'Exception' in val:
        return False
    return True


def _is_valid_genai(row, col_start, col_count=2):
    """检查该行中某个GenAI模型结果是否有效（情感值为三者之一）"""
    if col_start is None or col_start + col_count > len(row):
        return False
    val = str(row[col_start]).strip() if row[col_start] else ''
    return val in ('积极', '消极', '中性')


# ==================== 主分析流程 ====================
def StartAnalysis(input_file, output_file, colnum=1,
                  Ali=True, Baidu=True, GenAI=False,
                  emojitreat='replace',
                  has_header=True,
                  resume=True,
                  genai_models='gpt-4o-mini',
                  genai_prompt=None,
                  genai_api_key=None,
                  genai_base_url=None,
                  genai_sleep=0.5):
    """
    整体情感分析处理，支持平台级别断点续传。

    断点续传逻辑 (resume=True):
      1. 读取已有输出文件的全部数据
      2. 逐行检查每个平台的结果是否有效:
         - 有效 → 保留已有结果，不重复调用API
         - 无效/缺失/错误 → 仅重新调用该平台
      3. 支持"列扩展"：已有文件只有 Ali+Baidu 列时，
         自动补充 GenAI 列而不重跑 Ali+Baidu
      4. 写入前创建 .bak 备份，成功后自动删除

    参数:
        input_file:     输入 CSV 文件路径
        output_file:    输出 CSV 文件路径
        colnum:         文本所在列号 (从1开始)
        Ali:            是否使用阿里云
        Baidu:          是否使用百度
        GenAI:          是否使用 GenAI
        emojitreat:     emoji 处理方式 ('replace' / 'delete')
        has_header:     输入文件是否包含表头行

        resume:         是否启用断点续传 (默认 True)
                        - True: 平台级别续传，仅重试失败/缺失的平台
                        - False: 忽略已有输出，从头开始

        genai_models:   GenAI 模型，支持:
                          - 单个字符串: 'gpt-4o-mini'
                          - 列表: ['gpt-4o-mini', 'deepseek-chat']

        genai_prompt:   GenAI prompt，支持:
                          - None: 使用默认 prompt
                          - str: 所有模型共用
                          - dict: 按模型名指定

        genai_api_key:  API Key
        genai_base_url: API 地址
        genai_sleep:    GenAI 请求间隔(秒)
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
    #  读取输入文件
    # ======================================================================
    with open(input_file, 'r', encoding='utf-8') as f:
        input_reader = csv.reader(f)
        all_input_rows = list(input_reader)

    if has_header:
        original_header = all_input_rows[0]
        input_data_rows = all_input_rows[1:]
    else:
        original_header = [f'col_{i + 1}' for i in range(len(all_input_rows[0]))]
        input_data_rows = all_input_rows

    total_data_rows = len(input_data_rows)
    original_col_count = len(original_header)
    expected_header = original_header + new_headers
    expected_col_count = len(expected_header)

    # ======================================================================
    #  计算各平台在输出行中的列位置
    # ======================================================================
    col_positions = {}  # key -> (start_col_index, col_count)
    current_pos = original_col_count

    if Ali:
        col_positions['ali'] = (current_pos, 5)
        current_pos += 5
    if Baidu:
        col_positions['baidu'] = (current_pos, 5)
        current_pos += 5
    if GenAI:
        for model_name, label in model_labels:
            col_positions[f'genai_{model_name}'] = (current_pos, 2)
            current_pos += 2

    # ======================================================================
    #  断点续传：读取已有输出文件
    # ======================================================================
    existing_data = []       # 已有的输出数据行 (不含表头)
    resume_mode = 'fresh'    # 'fresh' | 'exact' | 'expand'
    backup_path = output_file + '.bak'

    if resume and os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        try:
            # 如果存在 .bak 文件（上次崩溃遗留），进行智能合并
            source_files = [output_file]
            if os.path.exists(backup_path):
                source_files.append(backup_path)
                print(f"  ℹ 检测到备份文件 {backup_path}，将合并恢复数据")

            # 读取所有源文件，取行数最多且列数兼容的
            best_data = []
            best_header = []
            for src in source_files:
                with open(src, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    try:
                        hdr = next(reader)
                        rows = list(reader)
                    except StopIteration:
                        continue
                    if len(rows) > len(best_data):
                        best_data = rows
                        best_header = hdr
                    elif len(rows) == len(best_data) and len(hdr) >= len(best_header):
                        best_data = rows
                        best_header = hdr

            existing_header = best_header
            existing_data = best_data

            # 判断列结构兼容性
            existing_col_count = len(existing_header)

            if existing_col_count == expected_col_count and existing_header == expected_header:
                # 完全匹配：平台级别续传
                resume_mode = 'exact'

            elif (existing_col_count < expected_col_count and
                  existing_header == expected_header[:existing_col_count]):
                # 已有文件是期望列的前缀（如之前没开 GenAI，现在加了）
                resume_mode = 'expand'
                # 补齐缺失列为空值
                pad_count = expected_col_count - existing_col_count
                existing_data = [row + [''] * pad_count for row in existing_data]

            elif existing_col_count <= expected_col_count:
                # 尝试按表头名称智能映射
                # 检查已有表头的原始数据列是否匹配
                if (existing_col_count >= original_col_count and
                        existing_header[:original_col_count] == expected_header[:original_col_count]):
                    resume_mode = 'expand'
                    # 重建行数据：保留原始列 + 按新列结构映射已有平台数据
                    mapped_data = []
                    for row in existing_data:
                        new_row = list(row[:original_col_count])  # 原始数据列
                        # 对每个平台，尝试从已有行中提取
                        for hdr_name in new_headers:
                            if hdr_name in existing_header:
                                idx = existing_header.index(hdr_name)
                                new_row.append(row[idx] if idx < len(row) else '')
                            else:
                                new_row.append('')
                        mapped_data.append(new_row)
                    existing_data = mapped_data
                else:
                    print(f"  ⚠ 已有输出文件列结构不兼容，将从头开始")
                    existing_data = []
                    resume_mode = 'fresh'
            else:
                print(f"  ⚠ 已有输出文件列数({existing_col_count})"
                      f"多于当前配置({expected_col_count})，将从头开始")
                existing_data = []
                resume_mode = 'fresh'

        except Exception as e:
            print(f"  ⚠ 读取已有输出文件失败({e})，将从头开始")
            existing_data = []
            resume_mode = 'fresh'

    # ======================================================================
    #  统计需要处理的工作量
    # ======================================================================
    stats = {
        'rows_fully_cached': 0,      # 所有平台都有效，无需API调用
        'rows_partial_cached': 0,    # 部分平台有效，需要补充
        'rows_fresh': 0,             # 无已有数据，需要全部调用
        'ali_cached': 0, 'ali_todo': 0,
        'baidu_cached': 0, 'baidu_todo': 0,
        'genai_cached': 0, 'genai_todo': 0,
    }

    for i in range(total_data_rows):
        if i < len(existing_data):
            row = existing_data[i]
            ali_ok = (not Ali) or _is_valid_ali(row, *col_positions['ali']) if Ali else True
            baidu_ok = (not Baidu) or _is_valid_baidu(row, *col_positions['baidu']) if Baidu else True
            genai_ok = True
            if GenAI:
                for model_name, label in model_labels:
                    if not _is_valid_genai(row, *col_positions[f'genai_{model_name}']):
                        genai_ok = False
                        break

            if ali_ok and baidu_ok and genai_ok:
                stats['rows_fully_cached'] += 1
            else:
                stats['rows_partial_cached'] += 1

            # 细粒度统计
            if Ali:
                if _is_valid_ali(row, *col_positions['ali']):
                    stats['ali_cached'] += 1
                else:
                    stats['ali_todo'] += 1
            if Baidu:
                if _is_valid_baidu(row, *col_positions['baidu']):
                    stats['baidu_cached'] += 1
                else:
                    stats['baidu_todo'] += 1
            if GenAI:
                for model_name, label in model_labels:
                    if _is_valid_genai(row, *col_positions[f'genai_{model_name}']):
                        stats['genai_cached'] += 1
                    else:
                        stats['genai_todo'] += 1
        else:
            stats['rows_fresh'] += 1
            if Ali:
                stats['ali_todo'] += 1
            if Baidu:
                stats['baidu_todo'] += 1
            if GenAI:
                stats['genai_todo'] += len(models_list)

    # 检查是否所有都已完成
    total_api_calls_needed = stats['ali_todo'] + stats['baidu_todo'] + stats['genai_todo']
    if total_api_calls_needed == 0 and stats['rows_fresh'] == 0:
        print("=" * 60)
        print(f"  ✓ 所有 {total_data_rows} 行、所有平台结果均有效，无需继续。")
        print(f"    如需强制重新分析，请设置 resume=False 或删除输出文件")
        print("=" * 60)
        # 清理遗留备份
        if os.path.exists(backup_path):
            os.remove(backup_path)
        return

    # ---- 打印分析配置 ----
    print("=" * 60)
    print("情感分析配置:")
    print(f"  输入文件: {input_file}")
    print(f"  输出文件: {output_file}")
    print(f"  文本列号: {colnum}")
    print(f"  总数据行: {total_data_rows}")
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
    print("-" * 60)
    print("  断点续传 (平台级别):")
    print(f"    续传模式: {resume_mode}")
    print(f"    完全缓存行 (无需API): {stats['rows_fully_cached']}")
    print(f"    部分缓存行 (需补充):  {stats['rows_partial_cached']}")
    print(f"    全新行 (需全部调用):   {stats['rows_fresh']}")
    if Ali:
        print(f"    阿里云 - 已缓存: {stats['ali_cached']}, 待调用: {stats['ali_todo']}")
    if Baidu:
        print(f"    百度   - 已缓存: {stats['baidu_cached']}, 待调用: {stats['baidu_todo']}")
    if GenAI:
        print(f"    GenAI  - 已缓存: {stats['genai_cached']}, 待调用: {stats['genai_todo']}")
    print(f"    总计需要API调用: {total_api_calls_needed} 次")
    print("=" * 60)

    # ======================================================================
    #  创建备份 (安全网)
    # ======================================================================
    if existing_data and os.path.exists(output_file):
        shutil.copy2(output_file, backup_path)
        print(f"  已创建备份: {backup_path}")

    # ======================================================================
    #  主处理循环
    # ======================================================================
    api_calls_made = 0
    rows_with_api_call = 0

    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(expected_header)
            outfile.flush()

            pbar = tqdm(total=total_data_rows, desc="Processing")

            for i, input_row in enumerate(input_data_rows):
                updated_row = list(input_row)

                # 获取文本
                try:
                    text = input_row[colnum - 1]
                except IndexError:
                    print(f"\n  警告：第 {colnum} 列不存在，跳过第 {i + 1} 行")
                    updated_row.extend([''] * len(new_headers))
                    writer.writerow(updated_row)
                    outfile.flush()
                    pbar.update(1)
                    continue

                # 空文本处理
                if not text.strip():
                    updated_row.extend([''] * len(new_headers))
                    writer.writerow(updated_row)
                    outfile.flush()
                    pbar.update(1)
                    continue

                # 获取已有数据 (如果有)
                existing_row = existing_data[i] if i < len(existing_data) else None
                row_made_api_call = False

                # ============================================================
                #  阿里云
                # ============================================================
                if Ali:
                    ali_start, ali_count = col_positions['ali']
                    if existing_row and _is_valid_ali(existing_row, ali_start, ali_count):
                        # 使用缓存结果
                        ali_values = existing_row[ali_start:ali_start + ali_count]
                    else:
                        # 需要调用API
                        try:
                            resultAli = SentimentAnalysisAli(text, emojitreat)
                        except Exception as e:
                            resultAli = {'AliCallError': str(e)}
                            print(f"\n  ⚠ 阿里云调用异常(行{i + 1}): {e}")
                        _, ali_values = _extract_ali_result(resultAli)
                        api_calls_made += 1
                        row_made_api_call = True
                    updated_row.extend(ali_values)

                # ============================================================
                #  百度
                # ============================================================
                if Baidu:
                    baidu_start, baidu_count = col_positions['baidu']
                    if existing_row and _is_valid_baidu(existing_row, baidu_start, baidu_count):
                        # 使用缓存结果
                        baidu_values = existing_row[baidu_start:baidu_start + baidu_count]
                    else:
                        # 需要调用API
                        try:
                            resultBaidu = SentimentAnalysisBaidu(text, emojitreat)
                        except Exception as e:
                            resultBaidu = {'BaiduCallError': str(e)}
                            print(f"\n  ⚠ 百度调用异常(行{i + 1}): {e}")
                        _, baidu_values = _extract_baidu_result(resultBaidu)
                        time.sleep(0.6)
                        api_calls_made += 1
                        row_made_api_call = True
                    updated_row.extend(baidu_values)

                # ============================================================
                #  GenAI (逐模型检查)
                # ============================================================
                if GenAI:
                    for model_name, label in model_labels:
                        genai_start, genai_count = col_positions[f'genai_{model_name}']
                        if existing_row and _is_valid_genai(existing_row, genai_start, genai_count):
                            # 使用缓存结果
                            genai_values = existing_row[genai_start:genai_start + genai_count]
                        else:
                            # 需要调用API
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
                                resultGenAI = {'success': False,
                                               'error': f'调用异常: {str(e)}',
                                               'model': model_name}
                                print(f"\n  ⚠ GenAI [{model_name}] 调用异常(行{i + 1}): {e}")
                            _, genai_values = _extract_genai_result(resultGenAI, label)
                            time.sleep(genai_sleep)
                            api_calls_made += 1
                            row_made_api_call = True
                        updated_row.extend(genai_values)

                # 写入该行
                writer.writerow(updated_row)
                outfile.flush()

                if row_made_api_call:
                    rows_with_api_call += 1

                pbar.update(1)

            pbar.close()

    except KeyboardInterrupt:
        print(f"\n\n  ⚠ 用户中断！已安全保存到第 {i} 行")
        print(f"    备份文件保留: {backup_path}")
        print(f"    下次运行将自动从中断处继续")
        return
    except Exception as e:
        print(f"\n\n  ✗ 发生异常: {e}")
        print(f"    已处理的数据已保存，备份文件保留: {backup_path}")
        print(f"    下次运行将自动从中断处继续")
        raise

    # ======================================================================
    #  完成：删除备份
    # ======================================================================
    if os.path.exists(backup_path):
        os.remove(backup_path)
        print(f"  已删除备份文件: {backup_path}")

    print(f"\n{'=' * 60}")
    print(f"分析完成！")
    print(f"  结果文件: {output_file}")
    print(f"  总行数: {total_data_rows}")
    print(f"  本次API调用: {api_calls_made} 次 (涉及 {rows_with_api_call} 行)")
    print(f"  缓存复用: {stats['rows_fully_cached']} 行完全复用")
    print(f"{'=' * 60}")


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
        output_file='../Results/Weibo216_multi_3agent.csv',
        colnum=3,
        Ali=True,
        Baidu=True,
        GenAI=True,
        emojitreat='replace',
        has_header=True,
        resume=True,
        genai_models=[ 'qwen3.6-flash', 'deepseek-v4-pro', 'claude-sonnet-4-6'],
        genai_prompt={ 'qwen3.6-flash': prompt_2,
                      'deepseek-v4-pro': prompt_2, 'claude-sonnet-4-6': prompt_2},
        genai_sleep=0.2
    )