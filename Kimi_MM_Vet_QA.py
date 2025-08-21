import os
import json
from transformers import AutoModelForCausalLM, AutoProcessor
from tqdm import tqdm  # 导入tqdm库
from PIL import Image
import torch

model_path = "moonshotai/Kimi-VL-A3B-Thinking-2506"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

results = []

# 读取JSON文件获取问题数据
json_path = "./MM-Vet/mm-vet_metadata.json"  # 输入JSON文件路径，请根据实际情况修改
with open(json_path, 'r', encoding='utf-8') as f:
    questions = json.load(f)[23:]

# 确保JSON数据是列表格式
if not isinstance(questions, list):
    raise ValueError("输入JSON文件内容应该是一个列表")

# 检查每个问题项是否包含必要的字段
required_fields = ['id', 'txt', 'img']
for idx, question in enumerate(questions):
    missing_fields = [field for field in required_fields if field not in question]
    if missing_fields:
        raise ValueError(f"问题项 {idx} 缺少必要的字段: {', '.join(missing_fields)}")

# 使用tqdm包装处理过程，显示进度
print("向模型提问并获取回答...")
for question in tqdm(questions, desc="处理问题"):
    question_id = question['id']
    text_prompt = question['txt']
    img_path = question['img']
    
    # 加载图片
    try:
        image = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"警告: 无法加载图片 {img_path}，错误信息: {str(e)}")
        # 可以选择跳过此问题或继续处理
        continue
    
    # 使用txt作为提示词，结合图片
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text_prompt}
            ]
        }
    ]
    
    # 处理输入并生成回答
    text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True, truncation=True).to(model.device)
    
    # 生成回答
    with torch.no_grad():  # 减少内存使用
        generated_ids = model.generate(** inputs, max_new_tokens=2048)
    
    # 提取生成的部分（去除输入部分）
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    # 解码生成的回答
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # 记录结果，包含id、txt、img和模型回答
    result = {
        "id": question_id,
        "txt": text_prompt,
        "img": img_path,
        "response": response
    }
    results.append(result)
    del inputs, generated_ids, generated_ids_trimmed
    torch.cuda.empty_cache()

# 保存结果到新的JSON文件
output_dir = "./MM-Vet"
os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
output_file = os.path.join(output_dir, "Kimi-VL-A3B-Thinking-2506.json")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"处理完成，结果已保存到 {output_file}")
    
