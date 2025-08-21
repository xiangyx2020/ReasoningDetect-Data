import os
import json
import pandas as pd
from transformers import Glm4vForConditionalGeneration, AutoProcessor
from tqdm import tqdm  # 导入tqdm库
from PIL import Image
import torch

model_path = "THUDM/GLM-4.1V-9B-Thinking"
model = Glm4vForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

results = []

# 读取CSV文件获取Behavior和BehaviorID
csv_path = "./S-Eval_Data/S-Eval_part3.csv"
csv_data = pd.read_csv(csv_path)

# 确保CSV有必要的列
required_columns = ['Behavior', 'BehaviorID']
if not all(col in csv_data.columns for col in required_columns):
    missing_cols = [col for col in required_columns if col not in csv_data.columns]
    raise ValueError(f"CSV文件中缺少必要的列: {', '.join(missing_cols)}")

# 提取Behavior和BehaviorID列表
behaviors = csv_data['Behavior'].tolist()
behavior_ids = csv_data['BehaviorID'].tolist()
total_items = len(behaviors)

# 获取selected_llava_images文件夹中的所有图片
image_dir = "./S-Eval_Part_3_LLaVa_Image/selected_llava_images"  # 图片文件夹路径
image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')
image_files = [
    f for f in os.listdir(image_dir)
    if os.path.isfile(os.path.join(image_dir, f))
    and f.lower().endswith(image_extensions)
]

# 检查图片数量是否足够
if len(image_files) < total_items:
    raise ValueError(f"图片数量不足，需要至少{total_items}张，实际只有{len(image_files)}张")

# 使用tqdm包装处理过程，显示进度
print("向模型提问并获取回答...")
for i in tqdm(range(total_items), desc="处理Behavior"):
    behavior = behaviors[i]
    behavior_id = behavior_ids[i]
    image_filename = image_files[i]  # 依次取出图片
    image_path = os.path.join(image_dir, image_filename)
    
    # 加载图片
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"警告: 无法加载图片 {image_path}，错误信息: {str(e)}")
        continue
    
    # 使用Behavior作为提示词，结合当前图片
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": behavior}
            ]
        }
    ]
    
    # 处理输入并生成回答
    text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True, truncation=True).to(model.device)
    
    # 生成回答，添加显存优化
    with torch.no_grad():
        generated_ids = model.generate(** inputs, max_new_tokens=2048)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # 记录结果，包含BehaviorID、使用的图片名和模型回答
    result = {
        "BehaviorID": behavior_id,
        "Behavior": behavior,
        "image_used": image_filename,  # 记录使用的图片名
        "response": response
    }
    results.append(result)
    
    # 清理显存
    del inputs, generated_ids, generated_ids_trimmed, image
    torch.cuda.empty_cache()

# 保存结果到JSON文件
output_dir = "./S-Eval_Part_3_LLaVa_Image"
os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
output_file = os.path.join(output_dir, "GLM-4.1V-9B-Thinking.json")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"处理完成，结果已保存到 {output_file}")
    