import json
import os
import shutil
from pathlib import Path

def process_json_and_copy_images(json_file_path, target_image_folder, base_image_dir=None):
    """
    处理JSON文件并复制对应图片
    
    参数:
        json_file_path: JSON文件的路径
        target_image_folder: 目标图片文件夹路径
        base_image_dir: 图片相对路径的基准目录，如果为None则使用当前工作目录
    """
    # 创建目标图片文件夹
    Path(target_image_folder).mkdir(parents=True, exist_ok=True)
    
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 确保数据是列表类型
    if not isinstance(data, list):
        raise ValueError("JSON文件内容应该是一个列表")
    
    # 取前100项
    top_100_items = data[:500]
    processed_items = []
    
    # 处理每一项
    for i, item in enumerate(top_100_items, 1):
        try:
            # 检查必要的键是否存在
            required_keys = ['question', 'image_relative_path', 'category', 'pair_type', 'response']
            for key in required_keys:
                if key not in item:
                    raise KeyError(f"第{i}项缺少必要的键: {key}")
            
            # 拆分think和response部分
            response = item['response']
            think_start = "<think>"
            think_end = "</think>"
            
            if think_start in response and think_end in response:
                # 提取思考部分
                think_content = response.split(think_start)[1].split(think_end)[0]
                # 提取剩余的response部分
                cleaned_response = response.split(think_end)[1] if len(response.split(think_end)) > 1 else ""
            else:
                # 如果没有思考标记，将整个response作为cleaned_response
                think_content = ""
                cleaned_response = response
                print(f"第{i}项没有包含完整的思考标记，已跳过拆分")
            
            # 存储处理后的内容（不包含原始response）
            processed_item = {
                'question': item['question'],
                'image_relative_path': item['image_relative_path'],
                'category': item['category'],
                'pair_type': item['pair_type'],
                'think': think_content,
                'cleaned_response': cleaned_response
            }
            processed_items.append(processed_item)
            
            # 处理图片复制
            image_path = item['image_relative_path']
            if image_path and image_path.strip():  # 检查路径是否非空
                # 构建完整的源图片路径
                if base_image_dir:
                    full_source_path = os.path.join(base_image_dir, image_path)
                else:
                    full_source_path = image_path
                
                # 检查源文件是否存在
                if os.path.exists(full_source_path) and os.path.isfile(full_source_path):
                    # 构建目标路径
                    image_filename = os.path.basename(image_path)
                    target_path = os.path.join(target_image_folder, image_filename)
                    
                    # 复制文件
                    shutil.copy2(full_source_path, target_path)
                    print(f"已复制第{i}项图片: {image_filename}")
                else:
                    print(f"第{i}项图片不存在: {full_source_path}")
            else:
                print(f"第{i}项图片路径为空，跳过复制")
                
        except Exception as e:
            print(f"处理第{i}项时出错: {str(e)}")
    
    # 将处理后的前100项保存为新的JSON文件
    with open('MM-SafetyBench-GLM.json', 'w', encoding='utf-8') as f:
        json.dump(processed_items, f, ensure_ascii=False, indent=2)
    print("已将处理后的前50项保存到 MM-SafetyBench-glm.json")

if __name__ == "__main__":
    # 配置参数 - 请根据实际情况修改以下路径
    JSON_FILE_PATH = "GLM-4.1V-9B-Thinking.json"  # 你的JSON文件路径
    TARGET_IMAGE_FOLDER = "MM-SafetyBench-50"  # 目标图片文件夹
    BASE_IMAGE_DIR = "."  # 图片相对路径的基准目录，如果图片路径是相对于JSON文件所在目录则保持"."
    
    # 执行处理
    process_json_and_copy_images(JSON_FILE_PATH, TARGET_IMAGE_FOLDER, BASE_IMAGE_DIR)