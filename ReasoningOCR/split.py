import json
import os

def process_json(input_file, output_file):
    """
    处理JSON文件：
    1. 将question字段移到最前面
    2. 将response字段拆分为think和cleaned_response
    3. 不保留原始的response字段
    4. 找不到think_end标记的项将被跳过，不保存到输出文件中
    """
    try:
        # 读取输入JSON文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processed_data = []
        skipped_missing_fields = 0  # 缺少必要字段的跳过计数
        skipped_missing_tags = 0    # 缺少标记的跳过计数
        
        for item in data:
            # 检查必要字段是否存在
            if "question" not in item or "response" not in item:
                print(f"警告：跳过无效条目（缺少必要字段）: {item.get('image_name', '未知图片')}")
                skipped_missing_fields += 1
                continue  # 跳过当前项，不添加到结果列表中
            
            # 拆分response字段
            response = item["response"]
            think_start = "◁think▷"
            think_end = "◁/think▷"            
            # 查找标记位置
            start_idx = response.find(think_start)
            end_idx = response.find(think_end)
            
            if start_idx == -1 or end_idx == -1:
                # 如果没有找到标记，跳过该条目，不保存到输出
                print(f"警告：条目 {item.get('image_name', '未知图片')} 缺少有效的标记，已跳过并不保存")
                skipped_missing_tags += 1
                continue  # 关键修改：跳过此条目，不添加到结果中
            else:
                # 提取思考过程和清洁后的响应
                think = response[start_idx + len(think_start) : end_idx].strip()
                cleaned_response = response[end_idx + len(think_end) :].strip()
            
            # 构建新的条目，确保question在最前面
            # 不包含原始的response字段
            processed_item = {
                "question": item["question"],
                "image_name": item.get("image_name", ""),
                "reasoning": think,
                "response": cleaned_response
            }

            processed_data.append(processed_item)
        
        # 保存处理后的JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        print(f"处理完成！共处理并保存 {len(processed_data)} 条有效数据，")
        print(f"其中：缺少必要字段的跳过 {skipped_missing_fields} 条，")
        print(f"缺少标记(think_start或think_end)的跳过 {skipped_missing_tags} 条，")
        print(f"结果已保存到 {output_file}")
        
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")

def main():
    # 输入和输出文件路径
    input_json = "Kimi-VL-A3B-Thinking-2506.json"    # 替换为你的输入文件路径
    output_json = "ReasoningOCR_Kimi.json"  # 替换为你的输出文件路径
    
    # 检查输入文件是否存在
    if not os.path.exists(input_json):
        print(f"错误：输入文件 {input_json} 不存在")
        return
    
    process_json(input_json, output_json)

if __name__ == "__main__":
    main()
