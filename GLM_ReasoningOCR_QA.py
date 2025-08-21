import os
import json
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
model.gradient_checkpointing_enable()
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

results = []

json_path = "./ReasoningOCR/qa.json"  
with open(json_path, 'r', encoding='utf-8') as f:
    questions = json.load(f) [123:]

if not isinstance(questions, list):
    raise ValueError("输入JSON文件内容应该是一个列表")

required_fields = ['q_id', 'img', 'question']  
for idx, question in enumerate(questions):
    missing_fields = [field for field in required_fields if field not in question]
    if missing_fields:
        raise ValueError(f"问题项 {idx} 缺少必要的字段: {', '.join(missing_fields)}")

print("向模型提问并获取回答...")
for question in tqdm(questions, desc="处理问题"):
    question_id = question['q_id'] 
    text_prompt = question['question'] 
    img_filename = question['img'] 
    
    img_path = os.path.join("./ReasoningOCR/img", img_filename)

    if not os.path.exists(img_path):
        print(f"警告: 图片文件不存在 {img_path}")
        continue

    try:
        image = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"警告: 无法加载图片 {img_path}，错误信息: {str(e)}")
        continue

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text_prompt}
            ]
        }
    ]

    text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True, truncation=True).to(model.device)

    with torch.no_grad():  
        generated_ids = model.generate(**inputs, max_new_tokens=2048)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    result = {
        "q_id": question_id,          
        "question": text_prompt,      
        "img": img_filename,          
        "response": response          
    }
    results.append(result)

    del inputs, generated_ids, generated_ids_trimmed
    torch.cuda.empty_cache()

output_dir = "./ReasoningOCR"
os.makedirs(output_dir, exist_ok=True) 
output_file = os.path.join(output_dir, "GLM-4.1V-result_3.json")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"处理完成，结果已保存到 {output_file}")
