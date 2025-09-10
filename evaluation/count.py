import json
from collections import defaultdict

def count_vote_patterns(input_file, output_file):
    """
    统计投票结果中三种票型的数量：
    - 3:0：全票通过（某一分数获得所有有效票）
    - 2:1：某一分数获得2票，另一分数获得1票
    - 1:1:1：三种分数各获得1票（仅当有效票数为3时）
    """
    try:
        # 读取之前生成的投票分布文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 初始化计数器
        pattern_counts = defaultdict(int)
        total_questions = len(data["各问题投票分布结果"])
        
        # 遍历每个问题的投票分布
        for question in data["各问题投票分布结果"]:
            vote_dist = question["投票分布"]
            valid_votes = question["有效票数"]
            
            # 提取各分数的得票数（按0、0.5、1排序）
            counts = [
                vote_dist["0分票数"],
                vote_dist["0.5分票数"],
                vote_dist["1分票数"]
            ]
            # 排序后便于判断票型（从高到低）
            sorted_counts = sorted(counts, reverse=True)
            
            # 判断票型
            if valid_votes == 3:
                if sorted_counts == [3, 0, 0]:
                    pattern_counts["3:0"] += 1
                elif sorted_counts == [2, 1, 0]:
                    pattern_counts["2:1"] += 1
                elif sorted_counts == [1, 1, 1]:
                    pattern_counts["1:1:1"] += 1
                else:
                    pattern_counts["其他（3票）"] += 1
            elif valid_votes == 2:
                if sorted_counts == [2, 0, 0]:
                    pattern_counts["3:0"] += 1  # 2:0也视为3:0类型的特殊情况
                elif sorted_counts == [1, 1, 0]:
                    pattern_counts["2:1"] += 1  # 1:1视为2:1类型的特殊情况
                else:
                    pattern_counts["其他（2票）"] += 1
            elif valid_votes == 1:
                pattern_counts["3:0"] += 1  # 1:0也视为3:0类型的特殊情况
            else:  # 0票
                pattern_counts["无有效票"] += 1
        
        # 生成统计结果
        result = {
            "统计说明": "统计三种主要票型的数量：\n"
                       "- 3:0：全票通过（某一分数获得所有有效票）\n"
                       "- 2:1：某一分数获得2票，另一分数获得1票（或有效票为2时1:1）\n"
                       "- 1:1:1：三种分数各获得1票（仅当有效票数为3时）",
            "总问题数": total_questions,
            "票型统计": dict(pattern_counts),
            "百分比": {
                pattern: f"{count/total_questions*100:.2f}%" 
                for pattern, count in pattern_counts.items()
            }
        }
        
        # 保存统计结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        
        print(f"票型统计完成！结果已保存至 {output_file}")
        print("统计结果概览：")
        for pattern, count in pattern_counts.items():
            print(f"- {pattern}：{count}个（{count/total_questions*100:.2f}%）")
            
    except Exception as e:
        print(f"统计过程中出错：{str(e)}")

if __name__ == "__main__":
    # 输入文件为之前生成的投票分布文件
    # 输出文件为票型统计结果
    count_vote_patterns(
        "./S-Eval_Part_1_FigStep/vote_distribution_GLM.json",
        "./S-Eval_Part_1_FigStep/vote_pattern_statistics_GLM.json"
    )
    