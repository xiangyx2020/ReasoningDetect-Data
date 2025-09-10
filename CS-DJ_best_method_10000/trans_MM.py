import os 
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langdetect import detect, LangDetectException
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def load_model_and_tokenizer(model_name="Qwen/Qwen3-8B"):
    """加载Qwen3-8B模型和分词器"""
    print(f"正在加载模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    # 确保tokenizer有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("模型加载完成")
    return model, tokenizer

def translate_text(text, model, tokenizer, target_lang="Chinese"):
    """使用Qwen3-8B将英文文本翻译为目标语言"""
    # 检查文本是否为英文
    try:
        if detect(text) != 'en':
            return text  # 非英文文本直接返回
    except LangDetectException:
        return text  # 无法检测语言时返回原文本
    
    # 构建翻译提示
    prompt = f"Translate the following English text to {target_lang} accurately, preserving the original meaning: {text}"
    messages = [{"role": "user", "content": prompt}]
    
    # 准备模型输入
    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    model_inputs = tokenizer([text_input], return_tensors="pt", padding=True, truncation=True).to(model.device)
    
    # 生成翻译结果
    with torch.no_grad():  # 禁用梯度计算以节省内存
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=2000,
            temperature=0.7,  # 控制输出随机性
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # 提取生成的内容
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    
    # 解析思考内容和实际翻译结果
    try:
        # 查找思考模式结束标记
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    
    # 获取并返回翻译内容
    translation = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return translation

def collect_strings(data, strings_list=None):
    """收集JSON中所有字符串用于进度显示"""
    if strings_list is None:
        strings_list = []
    
    if isinstance(data, dict):
        for v in data.values():
            collect_strings(v, strings_list)
    elif isinstance(data, list):
        for item in data:
            collect_strings(item, strings_list)
    elif isinstance(data, str) and len(data.strip()) > 0:
        strings_list.append(data)
    
    return strings_list

def process_json_data_with_progress(data, model, tokenizer, progress_bar):
    """带进度条的递归JSON处理函数"""
    if isinstance(data, dict):
        return {k: process_json_data_with_progress(v, model, tokenizer, progress_bar) 
                for k, v in data.items()}
    elif isinstance(data, list):
        return [process_json_data_with_progress(item, model, tokenizer, progress_bar) 
                for item in data]
    elif isinstance(data, str) and len(data.strip()) > 0:
        # 翻译并更新进度条
        result = translate_text(data, model, tokenizer)
        progress_bar.update(1)
        return result
    else:
        return data

def translate_json_file(input_path, output_path, model, tokenizer):
    """翻译JSON文件中的所有英文内容"""
    try:
        # 读取JSON文件
        print(f"读取文件: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 收集所有字符串以确定总进度
        print("分析文件内容...")
        total_strings = len(collect_strings(data))
        print(f"发现 {total_strings} 个文本片段需要处理")
        
        # 创建进度条
        with tqdm(total=total_strings, desc="翻译进度") as pbar:
            # 处理数据
            print("开始翻译...")
            translated_data = process_json_data_with_progress(data, model, tokenizer, pbar)
        
        # 写入翻译后的JSON文件
        print(f"保存结果到: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)
        
        print("翻译完成!")
        return True
    
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
        return False

def main():
    # 配置文件路径
    input_json_path = "mistral-3.2_eval_glm.json"    # 输入JSON文件路径
    output_json_path = "mistral-3.2_eval_glm_translated.json"  # 输出翻译后的JSON文件路径
    
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer()
    
    # 执行翻译
    translate_json_file(input_json_path, output_json_path, model, tokenizer)

if __name__ == "__main__":
    main()
