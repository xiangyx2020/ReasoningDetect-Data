import os
import pandas as pd
from datasets import load_dataset
from PIL import Image

def extract_step5_data(dataset_name="Monosail/HADES", output_image_dir="step5_images", output_csv="step5_texts.csv"):
    # 创建图片保存目录
    os.makedirs(output_image_dir, exist_ok=True)
    
    # 加载数据集
    print("加载数据集...")
    ds = load_dataset(dataset_name)
    
    # 确保test子集存在
    if "test" not in ds:
        raise ValueError("数据集中不存在'test'子集")
    
    # 筛选出step为5的样本
    print("筛选step为5的样本...")
    step5_data = ds["test"].filter(lambda x: x["step"] == 5)
    print(f"找到 {len(step5_data)} 个step为5的样本")
    
    if len(step5_data) == 0:
        print("没有找到step为5的样本")
        return
    
    # 提取文字信息并保存为CSV
    print("保存文字信息到CSV...")
    # 转换为DataFrame，排除image列
    text_columns = ['id', 'scenario', 'keywords', 'step', 'category', 'instruction']
    text_df = step5_data.to_pandas()[text_columns]
    text_df.to_csv(output_csv, index=False)
    
    # 保存图片（使用enumerate跟踪进度）
    print("保存图片...")
    for idx, item in enumerate(step5_data):
        # 获取图片和id
        image = item["image"]
        img_id = item["id"]
        
        # 确保id适合作为文件名（替换可能的非法字符）
        safe_id = str(img_id).replace("/", "_").replace("\\", "_").replace(":", "_")
        img_path = os.path.join(output_image_dir, f"{safe_id}.png")
        
        # 保存图片
        image.save(img_path)
        
        # 打印进度（每100张图片）
        if (idx + 1) % 100 == 0:
            print(f"已保存 {idx + 1}/{len(step5_data)} 张图片")
    
    print("提取完成！")
    print(f"图片已保存到: {os.path.abspath(output_image_dir)}")
    print(f"文字信息已保存到: {os.path.abspath(output_csv)}")

if __name__ == "__main__":
    # 可以根据需要修改参数
    extract_step5_data(
        dataset_name="Monosail/HADES",
        output_image_dir="./HADES/step5_images",
        output_csv="./HADES/step5_texts.csv"
    )
    