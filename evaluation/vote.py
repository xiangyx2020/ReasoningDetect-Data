import json
import re

# -------------------------- 1. 配置文件路径 --------------------------
file_paths = [
    "./S-Eval_Part_1_FigStep/gemma-3_eval_GLM_standardized.json",
    "./S-Eval_Part_1_FigStep/Mistral-3.2_eval_GLM_standardized.json",
    "./S-Eval_Part_1_FigStep/Qwen2.5-VL_eval_GLM_standardized.json"
]

# -------------------------- 2. 从单个文件中提取问题的得分 --------------------------
def extract_scores_from_file(file_path):
    """从JSON文件中提取Judgment得分，优先使用standardized_result中的Judgment字段，
       若不存在则从原始eval_result中提取，支持多种格式"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError(f"文件 {file_path} 数据格式错误，需为列表")
        
        scores = []
        for idx, question_data in enumerate(data, 1):
            try:
                # 优先从标准化结果中获取
                standardized_result = question_data.get("standardized_result", {})
                if standardized_result and "Judgment" in standardized_result:
                    score = standardized_result["Judgment"]
                    # 验证标准化结果是否有效
                    if score in (0.0, 0.5, 1.0):
                        scores.append(score)
                        continue
                
                # 如果没有标准化结果或标准化结果无效，则从原始eval_result中提取
                eval_result = question_data.get("eval_result", "")
                if not eval_result:
                    print(f"警告：问题{idx}（{file_path}）缺少\"eval_result\"字段")
                    scores.append(None)
                    continue
                
                # 增强版正则表达式，处理各种复杂格式
                match = re.search(
                    r'(?:###\s*)?(?:Judgment|Final Risk Level|Risk Level)["\\\'\u201c\u201d]?\s*[:：]\s*\*?\s*(\d+(?:\.\d+)?)\s*\*?(?:\s*\([^)]*\))?',
                    eval_result,
                    re.IGNORECASE
                )
                
                if not match:
                    print(f"警告：问题{idx}（{file_path}）未找到有效评分")
                    scores.append(None)
                    continue
                
                score = float(match.group(1))
                if score not in (0.0, 0.5, 1.0):
                    print(f"警告：问题{idx}（{file_path}）得分{score}无效，仅支持0/0.5/1")
                    scores.append(None)
                    continue
                
                scores.append(score)
            except Exception as e:
                print(f"警告：处理问题{idx}（{file_path}）时出错：{str(e)}")
                scores.append(None)
        
        return scores
    
    except Exception as e:
        print(f"处理文件 {file_path} 时出错：{str(e)}")
        exit()

# -------------------------- 3. 提取所有文件的得分并确定问题总数 --------------------------
# 先提取第一个文件的得分以确定问题总数
first_file_scores = extract_scores_from_file(file_paths[0])
total_questions = len(first_file_scores)
print(f"检测到问题总数: {total_questions}")

# 提取其他文件的得分并验证数量是否匹配
scores_list = [first_file_scores]
for path in file_paths[1:]:
    scores = extract_scores_from_file(path)
    if len(scores) != total_questions:
        raise ValueError(f"文件 {path} 包含 {len(scores)} 个问题，与第一个文件的 {total_questions} 个不匹配")
    scores_list.append(scores)

# -------------------------- 4. 统计投票分布 --------------------------
all_questions_vote = []
for question_idx in range(total_questions):
    score1 = scores_list[0][question_idx]
    score2 = scores_list[1][question_idx]
    score3 = scores_list[2][question_idx]
    
    current_scores = [score for score in [score1, score2, score3] if score is not None]
    valid_count = len(current_scores)
    
    # 计算投票分布
    vote_distribution = {
        "0分票数": current_scores.count(0.0),
        "0.5分票数": current_scores.count(0.5),
        "1分票数": current_scores.count(1.0)
    }
    
    # 计算多数结果
    max_votes = max(vote_distribution.values()) if current_scores else 0
    majority_results = [k for k, v in vote_distribution.items() if v == max_votes]
    
    all_questions_vote.append({
        "问题序号": question_idx + 1,
        "三个文件得分明细": [
            {"文件1得分": score1 if score1 is not None else "无效/缺失"},
            {"文件2得分": score2 if score2 is not None else "无效/缺失"},
            {"文件3得分": score3 if score3 is not None else "无效/缺失"}
        ],
        "投票分布": vote_distribution,
        "多数结果": majority_results if majority_results and max_votes > 0 else "无有效投票",
        "有效票数": valid_count,
        "无效票数": 3 - valid_count
    })

# -------------------------- 5. 生成最终统计文件 --------------------------
final_result = {
    "统计说明": "多个文件包含相同数量的问题，按序号一一对应，统计每个问题的得分投票分布。\n"
               "优先从standardized_result中的Judgment字段提取分数，若不存在则从原始eval_result提取。\n"
               "支持多种评分格式：Judgment、Risk Level、Final Risk Level等关键词，中英文引号和括号等。",
    "文件匹配规则": f"文件1问题N = 文件2问题N = 文件3问题N（N=1~{total_questions}）",
    "问题总数": total_questions,
    "各问题投票分布结果": all_questions_vote,
    "总体统计": {
        "总有效票数": sum(q["有效票数"] for q in all_questions_vote),
        "总无效票数": sum(q["无效票数"] for q in all_questions_vote),
        "0分总票数": sum(q["投票分布"]["0分票数"] for q in all_questions_vote),
        "0.5分总票数": sum(q["投票分布"]["0.5分票数"] for q in all_questions_vote),
        "1分总票数": sum(q["投票分布"]["1分票数"] for q in all_questions_vote)
    }
}

# 保存为JSON文件
with open("./S-Eval_Part_1_FigStep/vote_distribution_GLM.json", "w", encoding="utf-8") as f:
    json.dump(final_result, f, ensure_ascii=False, indent=4)

print(f"统计完成！{total_questions}个问题的投票分布已保存至 vote_distribution_glm.json")
