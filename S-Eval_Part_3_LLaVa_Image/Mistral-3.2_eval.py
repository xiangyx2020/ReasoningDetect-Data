import os
import json
from PIL import Image
import torch
from datetime import datetime, timedelta
import pathlib
from tqdm import tqdm

from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from huggingface_hub import hf_hub_download
from transformers import Mistral3ForConditionalGeneration

# 设置可见GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"
# 启用TensorFloat32提升性能（针对支持的GPU）
torch.set_float32_matmul_precision('high')

def load_image(image_path):
    """加载本地图片并返回PIL Image对象"""
    try:
        if os.path.exists(image_path):
            return Image.open(image_path).convert("RGB")
        else:
            print(f"图片路径不存在: {image_path}")
            return None
    except Exception as e:
        print(f"加载图片失败 {image_path}: {str(e)}")
        return None

def get_file_url(path):
    """将本地文件路径转换为file://格式的URL"""
    absolute_path = os.path.abspath(path)
    return pathlib.Path(absolute_path).as_uri()

def load_system_prompt(file_path):
    """从文件加载系统提示内容"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def load_examples(example_json_path, image_dir):
    """加载示例JSON文件，返回前3条有效示例"""
    try:
        with open(example_json_path, "r", encoding="utf-8") as f:
            raw_examples = json.load(f)
        
        processed_examples = []
        # 遍历并筛选有效示例
        for idx, example in enumerate(raw_examples):
            if len(processed_examples) >= 3:  # 仅保留前3条示例
                break
            
            # 校验示例的关键字段
            required_fields = ["image_relative_path", "query", "thinking", "response", "eval_result"]
            if not all(field in example for field in required_fields):
                print(f"跳过无效示例（缺失关键字段）: 第{idx+1}条")
                continue
            
            # 处理示例图片
            example_image_path = os.path.join(image_dir, example["image_relative_path"])
            example_image = load_image(example_image_path)
            
            # 封装示例数据
            processed_examples.append({
                "image_relative_path": example["image_relative_path"],
                "image": example_image,
                "query": example["query"],
                "thinking": example["thinking"],
                "response": example["response"],
                "eval_result": example["eval_result"]
            })
        
        if len(processed_examples) < 3:
            print(f"警告：仅加载到{len(processed_examples)}条有效示例（期望3条）")
        return processed_examples
    except Exception as e:
        print(f"加载示例JSON失败: {str(e)}")
        return []

def format_examples_as_dialog(examples):
    """将示例格式化为对话形式"""
    dialog_parts = [
        "Below are 3 reference examples, including the query, thinking process, final response, and evaluation result. Please strictly imitate the output style, ethical stance, and format of the examples for processing:\n"
    ]
    
    for idx, example in enumerate(examples, 1):
        dialog_parts.append(f"\n=== example{idx} ===")
        dialog_parts.append(f"image_path：{example['image_relative_path']}")
        dialog_parts.append(f"1.Query：{example['query']}")
        dialog_parts.append(f"2.model_thinking：{example['thinking']}")
        dialog_parts.append(f"3.model_response：{example['response']}")
        dialog_parts.append(f"4.eval_results：{example['eval_result']}")
    
    dialog_parts.append("\n--- Please process the following current task based on the above examples. ---")
    return "\n".join(dialog_parts)

def process_json_data(input_json, output_json, system_prompt, image_dir, model, tokenizer, examples):
    """处理JSON数据，调用Mistral模型生成结果并保存（带示例上下文）"""
    # 读取输入JSON文件
    with open(input_json, "r", encoding="utf-8") as f:
        data_items = json.load(f) 
    
    results = []
    device = model.device  # 获取模型所在设备
    example_dialog = format_examples_as_dialog(examples)  # 预处理示例为对话格式
    
    # 带进度条处理每个条目
    for item in tqdm(data_items, desc="Mistral-3.2 处理进度", unit="条"):
        try:
            # 获取图片名称并构建完整路径
            image_name = item.get("image_used")
            image_path = os.path.join(image_dir, image_name) if image_name else None
            
            # 加载图片
            image = None
            if image_name and image_path and os.path.exists(image_path):
                image = load_image(image_path)
            
            # 构建消息内容
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": []
                }
            ]
            
            # 添加示例对话上下文
            messages[1]["content"].append({"type": "text", "text": example_dialog})
            
            # 添加图片（如果存在）
            if image:
                # 修复图片URL格式
                image_url = get_file_url(image_path)
                messages[1]["content"].append({"type": "image_url", "image_url": {"url": image_url}})
            else:
                missing_note = "注意：该查询缺少相关图片或图片无法访问。"
                print(f"缺少图片或图片无法访问: {image_name or '未提供图片名称'}")
                messages[1]["content"].append({"type": "text", "text": missing_note})
            
            # 添加文本内容
            text_content = (
                f"Current Task Query:{item.get('Behavior', '无提问')}\n"
                f"Current Task Thinking:{item.get('think', '无思考过程')}\n"
                f"Current Task Cleaned Response:{item.get('cleaned_response', '无原始响应')}"
            )
            messages[1]["content"].append({"type": "text", "text": text_content})
            
            # 处理输入 - 使用Mistral的方式编码
            tokenized = tokenizer.encode_chat_completion(ChatCompletionRequest(messages=messages))
            
            input_ids = torch.tensor([tokenized.tokens]).to(device)
            attention_mask = torch.ones_like(input_ids).to(device)
            
            # 处理图像输入
            pixel_values = None
            image_sizes = None
            if image:
                pixel_values = torch.tensor(tokenized.images[0], dtype=torch.bfloat16).unsqueeze(0).to(device)
                image_sizes = torch.tensor([pixel_values.shape[-2:]]).to(device)
            
            # 生成结果
            with torch.inference_mode():
                generation = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_sizes=image_sizes,
                    max_new_tokens=1000,
                )[0]
            
            # 解码结果 - 移除skip_special_tokens参数，因为MistralTokenizer不支持
            eval_result = tokenizer.decode(generation[len(tokenized.tokens):])
            
            # 准备结果条目
            result_entry = {
                "image_name": image_name or "无图片名称",
                "BehaviorID": item.get("BehaviorID", ""),
                "query": item.get("Behavior", ""),
                "thinking": item.get("think", ""),
                "response": item.get("cleaned_response", ""),
                "eval_result": eval_result
            }
            
            if not image:
                result_entry["note"] += " | 该条目处理时缺少图片或图片无法访问"
                
            results.append(result_entry)
            
        except Exception as e:
            error_msg = f"处理 {item.get('image_used', '未知条目')} 时出错: {str(e)}"
            print(f"\n{error_msg}")
            results.append({
                "image_name": item.get("image_used", "未知图片"),
                "error": error_msg,
                "note": "处理失败（已跳过）"
            })
    
    # 保存结果
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    success_count = len([r for r in results if "error" not in r])
    fail_count = len(results) - success_count
    print(f"\n所有处理完成，结果已保存到 {output_json}")
    print(f"成功处理：{success_count}条 | 失败：{fail_count}条")

def main():
    # 配置文件路径
    CONFIG = {
        "system_prompt_file": "system_prompt_1.txt",
        "input_json_file": "S-Eval_3_GLM.json",
        "output_json_file": "Mistral-3.2_eval_GLM.json",
        "image_directory": "./selected_llava_images",
        "example_json_file": "icl_example.json",  # 示例文件路径
        "model_id": "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    }
    
    # 加载系统提示
    try:
        system_prompt = load_system_prompt(CONFIG["system_prompt_file"])
        print("✅ 系统提示加载完成")
    except Exception as e:
        print(f"❌ 加载系统提示失败: {str(e)}")
        return
    
    # 加载3条参考示例
    examples = load_examples(CONFIG["example_json_file"], CONFIG["image_directory"])
    if not examples:
        print("❌ 未加载到任何有效示例，无法进行示例引导，程序退出")
        return
    print(f"✅ 加载到{len(examples)}条有效参考示例")
    
    # 加载模型和分词器
    print(f"\n🔄 正在加载{CONFIG['model_id']}模型和分词器...")
    try:
        # 加载分词器
        tokenizer = MistralTokenizer.from_hf_hub(CONFIG["model_id"])
        
        # 加载模型
        model = Mistral3ForConditionalGeneration.from_pretrained(
            CONFIG["model_id"], 
            torch_dtype=torch.bfloat16,
            device_map="auto",  # 自动分配到可用GPU
            low_cpu_mem_usage=True  # 减少CPU内存占用
        ).eval()
        
        print(f"✅ 模型和分词器加载完成，模型位于: {model.device}")
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        return
    
    # 处理数据
    print("\n🚀 开始处理主数据...")
    process_json_data(
        input_json=CONFIG["input_json_file"],
        output_json=CONFIG["output_json_file"],
        system_prompt=system_prompt,
        image_dir=CONFIG["image_directory"],
        model=model,
        tokenizer=tokenizer,
        examples=examples
    )

if __name__ == "__main__":
    main()
