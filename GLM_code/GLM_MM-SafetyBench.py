import os
import json
from PIL import Image
from transformers import Glm4vForConditionalGeneration, AutoProcessor
from tqdm import tqdm

# 配置路径
model_path = "THUDM/GLM-4.1V-9B-Thinking"
json_file_path = "./MM-SafetyBench/selected_cases.json"  # 替换为实际的JSON文件路径
image_folder = "./MM-SafetyBench"      # 替换为实际的图片文件夹路径
output_json_path = "./MM-SafetyBench/GLM-4.1V-9B-Thinking.json"

# 加载模型和处理器
model = Glm4vForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# 读取JSON文件
with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

results = []

# 处理每个条目
for item in tqdm(data, desc="处理数据条目"):
    try:
        # 提取基础信息
        question = item["question"]
        pair_type = item["pair_type"]
        image_relative_path = item["image_relative_path"]
        category = item["category"]
        
        # 准备图像（如果需要）
        image = None
        if pair_type != "TEXT" and image_relative_path:
            image_path = os.path.join(image_folder, image_relative_path)
            if os.path.exists(image_path):
                image = Image.open(image_path)
            else:
                print(f"警告：图片文件不存在 - {image_path}")
        
        # 构建对话消息
        messages = [
            {
                "role": "user",
                "content": []
            }
        ]
        
        # 添加图像（如果有）
        if image is not None:
            messages[0]["content"].append({"type": "image", "image": image_path})
        
        # 添加文本问题
        messages[0]["content"].append({"type": "text", "text": question})
        
        # 处理对话模板
        text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        
        # 准备模型输入
        inputs_kwargs = {
            "text": text,
            "return_tensors": "pt",
            "padding": True,
            "truncation": True
        }
        if image is not None:
            inputs_kwargs["images"] = [image]
        
        inputs = processor(** inputs_kwargs).to(model.device)
        
        # 生成模型响应
        generated_ids = model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # 构建结果（包含原始信息和响应）
        result = item.copy()
        result["response"] = response
        results.append(result)
        
    except Exception as e:
        print(f"处理条目时出错：{str(e)}")
        # 出错时也保存原始信息，标记错误
        error_result = item.copy()
        error_result["response"] = f"处理出错：{str(e)}"
        results.append(error_result)

# 保存结果到新的JSON文件
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"处理完成，结果已保存到 {output_json_path}")
