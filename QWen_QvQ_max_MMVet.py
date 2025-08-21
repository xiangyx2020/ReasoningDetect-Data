import os
from openai import OpenAI
import base64
import json
from tqdm import tqdm

def encode_image(image_path):
    """将图片编码为base64格式"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# 配置参数
API_KEY = "sk-68b1fdd1be434956b901b91c8fd8996e"  # 替换为你的API Key
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
BASE_IMAGE_FOLDER = ""  # 基础图片文件夹路径，用于拼接相对路径
INPUT_JSON = "./MM-Vet/mm-vet_metadata.json"  # 输入的JSON文件路径，包含图文对
OUTPUT_FILE = "./MM-Vet/QWen_QWQ_max.json"  # 输出结果文件

# 初始化OpenAI客户端
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

# 读取输入JSON文件
try:
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data_entries = json.load(f)
    # 验证JSON结构
    required_keys = ["id", "txt", "img"]
    for entry in data_entries:
        for key in required_keys:
            if key not in entry:
                raise ValueError(f"输入JSON中缺少必要字段: {key}，在条目 {entry.get('id', '未知ID')} 中")
    print(f"成功读取输入JSON，共包含 {len(data_entries)} 个图文对")
except Exception as e:
    print(f"读取输入JSON失败: {e}")
    exit(1)

# 存储结果的列表
results = []

print("开始处理图文对...")
for entry in tqdm(data_entries, desc="处理进度", unit="个"):
    try:
        # 获取当前条目的信息
        entry_id = entry["id"]
        text_prompt = entry["txt"]
        img_relative_path = entry["img"]
        full_image_path = os.path.join(BASE_IMAGE_FOLDER, img_relative_path)

        # 检查图片文件是否存在
        if not os.path.exists(full_image_path):
            print(f"警告: 图片文件不存在 - {full_image_path}，跳过该条目")
            results.append({
                "id": entry_id,
                "txt": text_prompt,
                "img": img_relative_path,
                "status": "error",
                "message": "图片文件不存在"
            })
            continue

        # 编码图片
        base64_image = encode_image(full_image_path)

        # 准备消息内容
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                    {"type": "text", "text": text_prompt},
                ],
            },
        ]

        # 调用API
        completion = client.chat.completions.create(
            model="qvq-max",
            messages=messages,
            stream=True,
        )

        reasoning_content = ""  # 完整思考过程
        answer_content = ""    # 完整回复
        is_answering = False   # 是否进入回复阶段

        for chunk in completion:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # 收集思考内容
            if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                reasoning_content += delta.reasoning_content

            # 收集回复内容
            if hasattr(delta, "content") and delta.content:
                is_answering = True
                answer_content += delta.content

        # 保存成功处理的结果
        results.append({
            "id": entry_id,
            "txt": text_prompt,
            "img": img_relative_path,
            "status": "success",
            "reasoning": reasoning_content,
            "response": answer_content
        })

    except Exception as e:
        print(f"处理条目 {entry.get('id', '未知ID')} 时出错: {e}")
        results.append({
            "id": entry.get("id", "未知ID"),
            "txt": entry.get("txt", ""),
            "img": entry.get("img", ""),
            "status": "error",
            "message": str(e)
        })

# 将结果保存为JSON文件
with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
    json.dump(results, outfile, ensure_ascii=False, indent=4)

print(f"处理完成！共处理 {len(data_entries)} 个条目，结果已保存至: {OUTPUT_FILE}")
