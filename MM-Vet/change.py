import json
import os

def modify_json_file(input_file, output_file=None):
    """
    修改JSON文件：
    1. 移除"img"字段值中的"data/"前缀
    2. 删除"toxicity"字段
    
    参数:
        input_file: 输入JSON文件路径
        output_file: 输出JSON文件路径，若为None则覆盖输入文件
    """
    # 如果未指定输出文件，则覆盖输入文件
    if output_file is None:
        output_file = input_file
    
    try:
        # 读取JSON文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 处理数据
        # 检查数据是否为列表
        if isinstance(data, list):
            for item in data:
                # 处理img字段
                if "img" in item:
                    img_path = item["img"]
                    # 移除"data/"前缀
                    if img_path.startswith("data/"):
                        item["img"] = img_path[5:]  # "data/"长度为5
                # 删除toxicity字段
                if "toxicity" in item:
                    del item["toxicity"]
        # 处理数据为字典的情况
        elif isinstance(data, dict):
            # 处理img字段
            if "img" in data:
                img_path = data["img"]
                if img_path.startswith("data/"):
                    data["img"] = img_path[5:]
            # 删除toxicity字段
            if "toxicity" in data:
                del data["toxicity"]
        else:
            print("JSON数据格式不支持，必须是列表或字典")
            return
        
        # 写入修改后的数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"文件已成功修改并保存到: {output_file}")
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
    except json.JSONDecodeError:
        print(f"错误: 文件 {input_file} 不是有效的JSON格式")
    except Exception as e:
        print(f"处理文件时发生错误: {str(e)}")

if __name__ == "__main__":
    # 示例用法
    input_json = "./mm-vet_metadata.json"   # 替换为你的输入JSON文件路径
    output_json = "./mm-vet_metadata_1.json" # 替换为你的输出JSON文件路径，或设为None覆盖输入文件
    
    modify_json_file(input_json, output_json)
