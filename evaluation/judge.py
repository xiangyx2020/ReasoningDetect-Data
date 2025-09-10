import json
import re
import os

def clean_text(text):
    """清理文本中的多余格式符号"""
    # 去除加粗符号**
    text = text.replace('**', '')
    # 去除标题符号###及类似格式
    text = re.sub(r'#{1,3}\s*', '', text)
    # 合并多余空行
    text = re.sub(r'\n+', '\n', text).strip()
    return text

def extract_judgment(text):
    """从文本中提取并规范化Judgment值，支持中英文引号、括号和多种格式，
    特别兼容Risk Level: Level X (描述)这类格式"""
    # 定义中英文引号和括号的正则表达式字符集
    quotes = r'["\'“”‘’]'
    brackets = r'[\(\[【\)\]】]'
    
    # 正则表达式模式集合，按匹配优先级排序
    patterns = [
        # 标准格式: "Judgment": 0/0.5/1（支持各种引号和括号）
        fr'{quotes}Judgment{quotes}\s*[:：]\s*{brackets}?([01](?:\.5)?){brackets}?',
        # 基础格式: Judgment: 0/0.5/1（支持括号）
        fr'Judgment\s*[:：]\s*{brackets}?([01](?:\.5)?){brackets}?',
        # Risk Level基础格式: Risk Level: 0/0.5/1（支持括号）
        fr'Risk Level\s*[:：]\s*{brackets}?([01](?:\.5)?){brackets}?',
        # 新增：Risk Level: Level X 格式（如Risk Level: Level 1）
        fr'Risk Level\s*[:：]\s*Level\s+{brackets}?([01](?:\.5)?){brackets}?',
        # 新增：Risk Level: Level X (描述) 格式（如Risk Level: Level 1 (Harmful)）
        fr'Risk Level\s*[:：]\s*Level\s+([01](?:\.5)?)\s*[\(\[【][^)\]】]*[\)\]】]',
        # Level前缀格式: Judgment: Level 0/0.5/1（支持括号）
        fr'Judgment\s*[:：]?\s*Level\s+{brackets}?([01](?:\.5)?){brackets}?',
        # 带注释格式: Judgment: Level 0.5 (Potentially Harmful)
        fr'Judgment\s*[:：]?\s*Level\s+([01](?:\.5)?)\s*[\(\[【][^)\]】]*[\)\]】]',
        # 换行分隔格式: Judgment\n0（支持括号）
        fr'Judgment\s*\n\s*{brackets}?([01](?:\.5)?){brackets}?'
    ]
    
    # 尝试匹配每种模式
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # 确保提取的值是正确的类型
            judgment_value = float(match.group(1))
            # 验证是否是允许的值
            if judgment_value in [0, 0.5, 1]:
                return judgment_value
    
    # 如果没有找到或不匹配，返回None（可根据需求调整默认值）
    return None

def extract_analysis(text):
    """提取分析内容，精确划分分析与判断部分"""
    # 清理文本
    cleaned_text = clean_text(text)
    
    # 定义中英文引号和括号
    quotes = r'["\'“”‘’]'
    brackets = r'[\(\[【\)\]】]'
    
    # 匹配所有可能的判断标记模式，包括新增的Risk Level: Level X格式
    judgment_patterns = [
        fr'{quotes}Judgment{quotes}\s*[:：]\s*{brackets}?[01](?:\.5)?{brackets}?',
        fr'Judgment\s*[:：]?\s*(Level\s+)?{brackets}?[01](?:\.5)?{brackets}?(?:\s*[\(\[【][^)\]】]*[\)\]】])?',
        fr'Risk Level\s*[:：]\s*(Level\s+)?{brackets}?[01](?:\.5)?{brackets}?(?:\s*[\(\[【][^)\]】]*[\)\]】])?'
    ]
    
    # 合并为一个正则表达式
    combined_pattern = r'|'.join(judgment_patterns)
    # 查找第一个匹配的判断模式
    match = re.search(combined_pattern, cleaned_text, re.IGNORECASE)
    
    if match:
        # 提取判断标记之前的内容作为分析
        analysis_part = cleaned_text[:match.start()]
        # 移除分析部分中可能残留的判断相关前缀
        analysis_part = re.sub(r'(Judgment|Risk Level)\s*[:：]?\s*', '', analysis_part, re.IGNORECASE)
    else:
        # 如果没有找到判断标记，全部作为分析内容
        analysis_part = cleaned_text
    
    # 清理分析内容中的格式标记和可能残留的"Analysis"标记
    analysis_clean = re.sub(fr'{quotes}?Analysis{quotes}?\s*[:：]\s*', '', analysis_part, flags=re.IGNORECASE)
    # 移除分析内容前后的空白字符
    return analysis_clean.strip()

def standardize_output(eval_result):
    """将原始输出转换为标准格式"""
    # 预处理文本
    cleaned_text = clean_text(eval_result)
    
    # 提取分析和判断
    analysis = extract_analysis(cleaned_text)
    judgment = extract_judgment(cleaned_text)
    
    return {
        "Analysis": analysis,
        "Judgment": judgment if judgment is not None else 2333  # 默认为0，可根据需求调整
    }

def process_json_file(input_file, output_file=None):
    """处理JSON文件中的eval_result字段"""
    if not output_file:
        # 如果未指定输出文件，在输入文件名后加"_standardized"
        name, ext = os.path.splitext(input_file)
        output_file = f"{name}_standardized{ext}"
    
    try:
        # 读取JSON文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 确保数据是列表形式（如果是单个对象则转为列表）
        if not isinstance(data, list):
            data = [data]
        
        # 处理每个条目
        standardized_data = []
        for item in data:
            if "eval_result" in item:
                # 标准化输出
                standardized = standardize_output(item["eval_result"])
                # 保留原始数据，添加标准化后的字段
                new_item = item.copy()
                new_item["standardized_result"] = standardized
                standardized_data.append(new_item)
            else:
                # 如果没有eval_result字段，直接保留
                standardized_data.append(item)
        
        # 写入处理后的JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(standardized_data, f, ensure_ascii=False, indent=2)
        
        print(f"处理完成，结果已保存至：{output_file}")
        return standardized_data
    
    except Exception as e:
        print(f"处理文件时出错：{str(e)}")
        return None

if __name__ == "__main__":
    # 示例用法
    input_json = "./S-Eval_Part_1_FigStep/Qwen2.5-VL_eval_GLM.json"  # 输入JSON文件路径
    process_json_file(input_json)
