import pandas as pd
import re

def extract_number(behavior_id):
    """从BehaviorID中提取数字部分"""
    match = re.search(r'S-Eval_(\d+)', behavior_id)
    if match:
        return int(match.group(1))
    return 0  # 处理异常情况

def split_and_sort_csv(input_file, output_prefix):
    # 1. 读取原始CSV文件
    df = pd.read_csv(input_file, encoding="utf-8")
    
    # 2. 验证是否包含所需列
    required_columns = ['Behavior', 'Category', 'Tags', 'ContextString', 'BehaviorID']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV文件缺少必要的列: {col}")
    
    # 3. 按Category分组，确保每个类别内部数据随机打乱（保证分配均匀）
    df = df.sample(frac=1, random_state=42)  # 全局随机打乱
    grouped = df.groupby("Category", group_keys=False)
    
    # 4. 定义3个空DataFrame用于存储拆分后的数据
    df1 = pd.DataFrame(columns=df.columns)
    df2 = pd.DataFrame(columns=df.columns)
    df3 = pd.DataFrame(columns=df.columns)
    
    # 5. 遍历每个Category分组，按比例分配到3个文件
    for _, group in grouped:
        # 对每个类别的数据再次随机打乱
        group = group.sample(frac=1, random_state=42)
        total = len(group)
        
        # 计算每个文件应分配的数量
        count1 = total // 3
        count2 = total // 3
        count3 = total - count1 - count2  # 剩余数据分配给第三个文件
        
        # 分配数据到3个DataFrame
        df1 = pd.concat([df1, group.iloc[:count1]], ignore_index=True)
        df2 = pd.concat([df2, group.iloc[count1:count1+count2]], ignore_index=True)
        df3 = pd.concat([df3, group.iloc[count1+count2:]], ignore_index=True)
    
    # 6. 为每个DataFrame添加数字列用于排序
    for df in [df1, df2, df3]:
        # 提取BehaviorID中的数字
        df['SortNumber'] = df['BehaviorID'].apply(extract_number)
        # 按数字排序
        df.sort_values(by='SortNumber', inplace=True)
        # 删除临时排序列
        df.drop(columns=['SortNumber'], inplace=True)
        # 重置索引
        df.reset_index(drop=True, inplace=True)
    
    # 7. 保存拆分并排序后的文件
    df1.to_csv(f"{output_prefix}_part1.csv", index=False, encoding="utf-8")
    df2.to_csv(f"{output_prefix}_part2.csv", index=False, encoding="utf-8")
    df3.to_csv(f"{output_prefix}_part3.csv", index=False, encoding="utf-8")
    
    print("拆分和排序完成！")
    print(f"文件1行数：{len(df1)}，类别分布：\n{df1['Category'].value_counts()}")
    print(f"文件2行数：{len(df2)}，类别分布：\n{df2['Category'].value_counts()}")
    print(f"文件3行数：{len(df3)}，类别分布：\n{df3['Category'].value_counts()}")

# 使用示例
if __name__ == "__main__":
    input_file = "./S-Eval_base_risk_en_medium.csv"  # 替换为你的输入文件路径
    output_prefix = "S-Eval"  # 输出文件前缀
    split_and_sort_csv(input_file, output_prefix)
