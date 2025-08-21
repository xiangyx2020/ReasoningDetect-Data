import os
import json
import pandas as pd
from transformers import AutoModelForCausalLM, AutoProcessor
from tqdm import tqdm  # 导入tqdm库

model_path = "moonshotai/Kimi-VL-A3B-Thinking-2506"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

results = []

# 读取CSV文件获取Behavior和BehaviorID
csv_path = "./S-Eval_Data/S-Eval_part2.csv"
csv_data = pd.read_csv(csv_path)

# 确保CSV有必要的列
required_columns = ['Behavior', 'BehaviorID']
if not all(col in csv_data.columns for col in required_columns):
    missing_cols = [col for col in required_columns if col not in csv_data.columns]
    raise ValueError(f"CSV文件中缺少必要的列: {', '.join(missing_cols)}")

# 提取Behavior和BehaviorID列表
behaviors = csv_data['Behavior'].tolist()
behavior_ids = csv_data['BehaviorID'].tolist()

# 使用tqdm包装处理过程，显示进度
print("向模型提问并获取回答...")
for i in tqdm(range(len(behaviors)), desc="处理Behavior"):
    behavior = behaviors[i]
    behavior_id = behavior_ids[i]
    
    # 使用Behavior作为提示词
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": None},
                {"type": "text", "text": behavior}
            ]
        }
    ]
    
    # 处理输入并生成回答
    text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True).to(model.device)
    generated_ids = model.generate(** inputs, max_new_tokens=2048)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # 记录结果，包含Behavior、BehaviorID和模型回答
    result = {
        "BehaviorID": behavior_id,
        "Behavior": behavior,
        "response": response
    }
    results.append(result)

# 保存结果到JSON文件
output_dir = "./S-Eval_Part_2_Text"
os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
with open(os.path.join(output_dir, "Kimi_VL_A3B_Thinking-2506.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"处理完成，结果已保存到 {os.path.join(output_dir, 'Kimi_VL_A3B_Thinking-2506.json')}")
