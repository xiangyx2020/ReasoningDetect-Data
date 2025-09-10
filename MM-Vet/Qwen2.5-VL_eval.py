import os
import json
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

# 设置可见GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7,8,9"
torch.set_float32_matmul_precision('high')

def load_image(image_path):
    """加载本地图片并返回PIL Image对象（适配示例与主数据的图片路径）"""
    try:
        if os.path.exists(image_path):
            return Image.open(image_path).convert("RGB")
        else:
            print(f"图片路径不存在: {image_path}")
            return None
    except Exception as e:
        print(f"加载图片失败 {image_path}: {str(e)}")
        return None

def load_system_prompt(file_path):
    """从文件加载系统提示内容"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def load_examples(example_json_path, image_dir):
    """加载示例JSON文件（严格匹配示例字段格式），返回前3条有效示例"""
    try:
        with open(example_json_path, "r", encoding="utf-8") as f:
            raw_examples = json.load(f)
        
        processed_examples = []
        # 遍历并筛选有效示例（确保包含关键字段）
        for idx, example in enumerate(raw_examples):
            if len(processed_examples) >= 3:  # 仅保留前3条示例
                break
            
            # 校验示例的关键字段（匹配用户提供的示例格式）
            required_fields = ["image_relative_path", "query", "thinking", "response", "eval_result"]
            if not all(field in example for field in required_fields):
                print(f"跳过无效示例（缺失关键字段）: 第{idx+1}条")
                continue
            
            # 处理示例图片（使用image_relative_path构建完整路径）
            example_image_path = os.path.join(image_dir, example["image_relative_path"])
            example_image = load_image(example_image_path)
            
            # 封装示例数据（与原始格式一致，便于后续格式化）
            processed_examples.append({
                "image_relative_path": example["image_relative_path"],
                "image": example_image,  # 加载后的图片对象
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
    """将示例格式化为“用户-助手”对话形式（符合Qwen模型对话逻辑）"""
    dialog_parts = [
        "Below are 3 reference examples, including the query, thinking process, final response, and evaluation result. Please strictly imitate the output style, ethical stance, and format of the examples for processing:\n"
    ]
    
    for idx, example in enumerate(examples, 1):
        # 示例标题（标注类别和图片信息）
        dialog_parts.append(f"\n=== example{idx} ===")
        dialog_parts.append(f"image_path：{example['image_relative_path']}")
        dialog_parts.append(f"1.Query：{example['query']}")
        dialog_parts.append(f"2.model_thinking：{example['thinking']}")
        dialog_parts.append(f"3.model_response：{example['response']}")
        dialog_parts.append(f"4.eval_results：{example['eval_result']}")
    
    # 引导模型处理当前任务
    dialog_parts.append("\n--- Please process the following current task based on the above examples. ---")
    return "\n".join(dialog_parts)

def process_json_data(input_json, output_json, system_prompt, image_dir, 
                     model, processor, examples):
    """处理主数据（整合示例对话上下文）"""
    # 读取主数据JSON
    with open(input_json, "r", encoding="utf-8") as f:
        data_items = json.load(f)
    
    results = []
    device = model.device
    # 预处理示例为对话格式（仅执行一次，避免重复计算）
    example_dialog = format_examples_as_dialog(examples)
    
    # 带进度条处理每个主数据条目
    for item in tqdm(data_items, desc="Qwen2.5-VL 处理进度", unit="条"):
        try:
            # 1. 处理当前条目的图片
            item_image_path = os.path.join(image_dir, item.get("image_name", ""))
            item_image = load_image(item_image_path) if item.get("image_name") else None
            
            # 2. 构建完整对话（系统提示 + 示例上下文 + 当前任务）
            messages = [
                # 系统提示（优先定义整体规则）
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}]
                },
                # 用户消息（包含示例上下文 + 当前任务的图片/文本）
                {
                    "role": "user",
                    "content": []
                }
            ]
            
            # 2.1 添加示例对话上下文（作为当前任务的前置参考）
            messages[1]["content"].append({"type": "text", "text": example_dialog})
            
            # 2.2 添加当前任务的图片（若存在）
            if item_image:
                messages[1]["content"].append({
                    "type": "image", 
                    "image": item_image,
                    "caption": f"当前任务图片：{item.get('image_name')}"  # 标注图片信息
                })
            else:
                messages[1]["content"].append({
                    "type": "text", 
                    "text": "当前任务缺少图片或图片无法访问"
                })
            
            # 2.3 添加当前任务的文本内容（匹配示例的query格式）
            item_text = (
                f"Current Task Query:{item.get('question','无提问')}\n"
                f"Current Task Thinking:{item.get('reasoning', '无思考过程')}\n"
                f"Current Task Cleaned Response:{item.get('response', '无原始响应')}"
            )
            messages[1]["content"].append({"type": "text", "text": item_text})
            
            # 3. Qwen模型输入处理（多模态格式适配）
            # 3.1 生成对话模板文本
            chat_text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True  # 自动添加生成提示
            )
            # 3.2 处理视觉信息（图片）
            image_inputs, video_inputs = process_vision_info(messages)
            # 3.3 编码为模型可接受的张量
            inputs = processor(
                text=[chat_text],
                images=image_inputs,
                videos=video_inputs,
                padding="longest",  # 适配多模态输入长度
                return_tensors="pt",
                truncation=True,  # 避免输入过长（Qwen有最大长度限制）
                max_length=32768  # 基于Qwen2.5-VL-32B的默认最大长度
            ).to(device)
            
            # 4. 模型生成（确定性生成，避免无效参数）
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=2000,  # 足够容纳思考+响应+评估结果
                    do_sample=False,  # 禁用采样，确保输出稳定（匹配示例风格）
                    temperature=None,  # 确定性生成时无需温度参数
                    top_p=None,  # 移除无效采样参数，避免报错
                    top_k=None
                )
            
            # 5. 结果解码与修剪（仅保留新生成部分）
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            # 解码时保留原始格式（不清理空格，匹配示例的换行结构）
            eval_result = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            # 6. 保存结果（与主数据格式一致，新增示例参考标记）
            result_entry = {
                "image_name": item.get("image_name", "无图片"),
                "query": item.get("question", ""),
                "thinking": item.get("reasoning", ""),
                "response": item.get("response", ""),
                "eval_result": eval_result
            }
            if not item_image:
                result_entry["note"] += " | 缺少当前任务图片"
            
            results.append(result_entry)
        
        except Exception as e:
            # 错误处理（保留错误信息，便于后续排查）
            error_msg = f"处理条目[{item.get('image_name', '未知')}]失败: {str(e)}"
            print(f"\n{error_msg}")
            results.append({
                "image_name": item.get("image_name", "未知图片"),
                "error": error_msg,
                "note": "处理失败（已跳过）"
            })
    
    # 保存最终结果
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n所有处理完成！结果已保存至：{output_json}")
    print(f"成功处理：{len([r for r in results if 'error' not in r])}条 | 失败：{len([r for r in results if 'error' in r])}条")

def main():
    # -------------------------- 配置参数 --------------------------
    CONFIG = {
        "system_prompt_file": "system_prompt_1.txt",  # 系统提示文件
        "input_json_file": "MMVet_Kimi.json",  # 主数据文件
        "output_json_file": "Qwen2.5-VL_eval_Kimi.json",  # 输出文件
        "image_directory": "./images",  # 图片根目录
        "example_json_file": "icl_example.json",  # 3条示例的JSON文件路径
        "model_id": "Qwen/Qwen2.5-VL-32B-Instruct"  # Qwen模型ID
    }
    # --------------------------------------------------------------
    
    # 1. 加载系统提示
    try:
        system_prompt = load_system_prompt(CONFIG["system_prompt_file"])
        print("✅ 系统提示加载完成")
    except Exception as e:
        print(f"❌ 加载系统提示失败: {e}")
        return
    
    # 2. 加载3条参考示例
    examples = load_examples(CONFIG["example_json_file"], CONFIG["image_directory"])
    if not examples:
        print("❌ 未加载到任何有效示例，无法进行示例引导，程序退出")
        return
    print(f"✅ 加载到{len(examples)}条有效参考示例")
    
    # 3. 加载Qwen模型和处理器
    print(f"\n🔄 正在加载模型：{CONFIG['model_id']}（请耐心等待）...")
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            CONFIG["model_id"],
            torch_dtype="auto",  # 自动匹配GPU支持的精度（如bfloat16）
            device_map="auto",  # 自动分配多GPU（适配6-9号GPU）
            low_cpu_mem_usage=True,  # 减少CPU内存占用，避免OOM
            trust_remote_code=True  # 加载Qwen的自定义代码（多模态处理需开启）
        ).eval()  # 切换为推理模式
        processor = AutoProcessor.from_pretrained(
            CONFIG["model_id"],
            trust_remote_code=True
        )
        print(f"✅ 模型加载完成，主设备：{model.device}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 4. 处理主数据（整合示例上下文）
    print("\n🚀 开始处理主数据...")
    process_json_data(
        input_json=CONFIG["input_json_file"],
        output_json=CONFIG["output_json_file"],
        system_prompt=system_prompt,
        image_dir=CONFIG["image_directory"],
        model=model,
        processor=processor,
        examples=examples
    )

if __name__ == "__main__":
    main()