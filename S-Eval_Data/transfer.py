import json
import csv

# 读取json文件内容
with open('./S-Eval_base_risk_en_medium.jsonl', 'r', encoding='utf-8') as json_file:
    json_data = json_file.readlines()

# 处理数据并准备写入csv
csv_data = []
for index, line in enumerate(json_data, 1):
    item = json.loads(line)
    behavior = item['prompt']
    category = item['risk_type']
    behavior_id = f"S-Eval_{index}"
    csv_data.append([behavior, category, '', '', behavior_id])

# 写入csv文件
with open('S-Eval_base_risk_en_medium.csv', 'w', encoding='utf-8', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Behavior', 'Category', 'Tags', 'ContextString', 'BehaviorID'])
    csv_writer.writerows(csv_data)