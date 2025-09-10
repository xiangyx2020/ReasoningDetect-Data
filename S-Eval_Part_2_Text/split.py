import json
import os

def split_response_field(json_data):
    """
    将每个条目里的response字段拆分为think和cleaned_response
    遇到异常时在命令行报错并指明无法处理的条目
    """
    # 确保输入是列表
    if not isinstance(json_data, list):
        raise ValueError("输入的JSON数据应该是一个列表")
    
    processed_data = []
    error_count = 0  # 记录错误条目数量
    
    for index, item in enumerate(json_data):
        # 检查必要字段是否存在
        if "BehaviorID" not in item:
            error_msg = f"条目索引 {index}: 缺少BehaviorID字段，无法标识该条目"
            print(f"错误: {error_msg}")
            error_count += 1
            processed_data.append(item)  # 保留原始条目
            continue
            
        behavior_id = item["BehaviorID"]
        
        if "response" not in item:
            error_msg = f"条目 BehaviorID: {behavior_id} (索引 {index}): 缺少response字段"
            print(f"错误: {error_msg}")
            error_count += 1
            processed_data.append(item)  # 保留原始条目
            continue
            
        response = item["response"]
        # 查找分隔符
        start_tag = "<think>"
        end_tag = "</think>"
        
        start_idx = response.find(start_tag)
        end_idx = response.find(end_tag)
        
        # 处理不同情况
        if start_idx == -1 or end_idx == -1:
            # 没有找到完整的标签，记录错误
            error_msg = f"条目 BehaviorID: {behavior_id} (索引 {index}): 缺少完整的标签"
            if start_idx == -1:
                error_msg += "，未找到◁think▷标签"
            if end_idx == -1:
                error_msg += "，未找到◁/think▷标签"
            print(f"错误: {error_msg}")
            error_count += 1
            
            # 将整个response作为cleaned_response，think设为空
            think = ""
            cleaned_response = response
        elif start_idx >= end_idx:
            # 标签顺序错误
            error_msg = f"条目 BehaviorID: {behavior_id} (索引 {index}): 标签顺序错误，◁think▷应在◁/think▷之前"
            print(f"错误: {error_msg}")
            error_count += 1
            
            # 仍然尝试提取内容
            think = ""
            cleaned_response = response
        else:
            # 正常提取内容
            think_start = start_idx + len(start_tag)
            think = response[think_start:end_idx].strip()
            
            cleaned_start = end_idx + len(end_tag)
            cleaned_response = response[cleaned_start:].strip()
        
        # 创建新条目，移除原response，添加新字段
        new_item = item.copy()
        del new_item["response"]
        new_item["think"] = think
        new_item["cleaned_response"] = cleaned_response
        
        processed_data.append(new_item)
    
    # 汇总错误信息
    if error_count > 0:
        print(f"\n处理完成，共发现 {error_count} 个异常条目，请查看上述错误信息")
    else:
        print("\n处理完成，所有条目均正常处理")
    
    return processed_data

def process_json_file(input_file, output_file):
    """处理JSON文件，将response字段拆分"""
    try:
        # 读取输入文件
        with open(input_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # 处理数据
        processed_data = split_response_field(json_data)
        
        # 写入输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=4)
        
        print(f"结果已保存到: {output_file}")
        return output_file
        
    except FileNotFoundError:
        print(f"错误: 找不到输入文件 '{input_file}'")
    except json.JSONDecodeError:
        print(f"错误: 输入文件 '{input_file}' 不是有效的JSON格式")
    except Exception as e:
        print(f"处理文件时出错: {str(e)}")
    return None

if __name__ == "__main__":
    # 预设JSON文件名称
    input_filename = "GLM-4.1V-9B-Thinking.json"    # 输入文件名称
    output_filename = "S-Eval_2_GLM.json"  # 输出文件名称
    
    # 处理文件
    process_json_file(input_filename, output_filename)
