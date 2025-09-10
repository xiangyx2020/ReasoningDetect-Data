import json
import os
import shutil
from PIL import Image
import requests
from io import BytesIO

def extract_3_0_votes(stat_file, qwen_file, output_json, image_folder, image_prefix="/images"):
    """
    提取投票比例为3:0的项目及其图片，图片检索添加共同前缀，仅保留原始内容和投票结果
    
    参数:
    stat_file: 统计结果JSON文件路径
    qwen_file: Qwen2.5-VL的JSON文件路径
    output_json: 输出结果JSON文件路径
    image_folder: 图片保存文件夹路径
    image_prefix: 图片检索时使用的共同前缀
    """
    # 创建图片保存文件夹
    os.makedirs(image_folder, exist_ok=True)
    
    # 读取统计结果
    with open(stat_file, 'r', encoding='utf-8') as f:
        stat_data = json.load(f)
    
    # 筛选出投票比例为3:0的项目并确定投票结果
    filtered_items = []
    for item in stat_data['各问题投票分布结果']:
        vote_dist = item['投票分布']
        # 检查是否有一个选项获得3票，其他为0
        if vote_dist['0分票数'] == 3:
            filtered_items.append({
                "question_idx": item['问题序号'] - 1,  # 0-based索引
                "vote_result": 0.0
            })
        elif vote_dist['0.5分票数'] == 3:
            filtered_items.append({
                "question_idx": item['问题序号'] - 1,  # 0-based索引
                "vote_result": 0.5
            })
        elif vote_dist['1分票数'] == 3:
            filtered_items.append({
                "question_idx": item['问题序号'] - 1,  # 0-based索引
                "vote_result": 1.0
            })
    
    print(f"共找到 {len(filtered_items)} 个投票比例为3:0的项目")
    
    # 读取Qwen2.5-VL的JSON文件
    with open(qwen_file, 'r', encoding='utf-8') as f:
        qwen_data = json.load(f)
        if not isinstance(qwen_data, list):
            raise ValueError("Qwen2.5-VL.json文件内容应为列表格式")
    
    # 收集结果并保存图片
    result = []
    for item in filtered_items:
        question_idx = item["question_idx"]
        vote_result = item["vote_result"]
        
        # 获取对应的Qwen数据
        if question_idx < 0 or question_idx >= len(qwen_data):
            print(f"警告：问题索引 {question_idx} 在Qwen文件中不存在")
            continue
            
        # 获取原始内容（保留原有字段）
        qwen_item = qwen_data[question_idx]
        original_content = {
            "image_name": qwen_item.get("image_name"),
            "query": qwen_item.get("query"),
            "thinking": qwen_item.get("thinking"),
            "response": qwen_item.get("response"),
            "eval_result": qwen_item.get("eval_result")
        }
        
        # 处理图片，添加共同的/images前缀进行检索
        image_name = qwen_item.get("image_name")
        if image_name:
            try:
                image_save_path = os.path.join(image_folder, image_name)
                
                # 构建带前缀的图片路径进行检索
                prefixed_image_path = os.path.join(image_prefix, image_name)
                
                # 查找图片的可能路径（包含带前缀的路径）
                possible_paths = [
                    image_name,  # 原始文件名
                    os.path.join(os.path.dirname(qwen_file), image_name),  # 与Qwen文件同目录
                    prefixed_image_path,  # 带前缀的路径
                    os.path.abspath(prefixed_image_path)  # 带前缀的绝对路径
                ]
                
                found = False
                for path in possible_paths:
                    if os.path.exists(path):
                        shutil.copy2(path, image_save_path)
                        found = True
                        break
                
                # 尝试URL下载（如果是网络地址）
                if not found and (image_name.startswith(('http://', 'https://'))):
                    response = requests.get(image_name, timeout=10)
                    response.raise_for_status()
                    with Image.open(BytesIO(response.content)) as img:
                        img.save(image_save_path)
                    found = True
                
                if not found:
                    print(f"警告：问题索引 {question_idx} 的图片在以下路径均未找到: {possible_paths}")
            
            except Exception as e:
                print(f"警告：保存问题索引 {question_idx} 的图片时出错: {str(e)}")
        
        # 只添加原始内容和投票结果
        result.append({
            **original_content,  # 展开原始内容字段
            "vote_result": vote_result  # 添加投票结果
        })
    
    # 保存筛选结果
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    
    print(f"处理完成！结果已保存至 {output_json}")
    print(f"图片已保存至 {image_folder} 文件夹")

if __name__ == "__main__":
    # 配置文件路径
    STATISTICS_FILE = "vote_distribution_Kimi.json"
    QWEN_FILE = "Qwen2.5-VL_eval_Kimi.json"
    OUTPUT_JSON = "3_0_votes_Kimi.json"
    IMAGE_FOLDER = "3_0_votes_images"
    IMAGE_PREFIX = "./images"  # 图片检索的共同前缀
    
    extract_3_0_votes(
        STATISTICS_FILE, 
        QWEN_FILE, 
        OUTPUT_JSON, 
        IMAGE_FOLDER,
        IMAGE_PREFIX
    )
    