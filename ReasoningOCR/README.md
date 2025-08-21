---
license: cc-by-sa-4.0
task_categories:
  - visual-question-answering
language:
  - en
  - zh
tags:
  - large-multimodal-models
  - logical-reasoning
  - OCR
  - domain-knowledge-free
size_categories:
  - n<1K
---

# Reasoning-OCR: Can Large Multimodal Models Solve Complex Logical Reasoning Problems from OCR Cues?

Large Multimodal Models (LMMs) have become increasingly versatile, accompanied by impressive Optical Character Recognition (OCR) related capabilities. 
Existing OCR-related benchmarks emphasize evaluating LMMs' abilities of relatively simple visual question answering, visual-text parsing, etc. However, the extent to which LMMs can deal with complex logical reasoning problems based on OCR cues is relatively unexplored.  
To this end, we introduce the **Reasoning-OCR** benchmark, which challenges LMMs to solve complex reasoning problems based on the cues that can be extracted from rich visual-text. 
**Reasoning-OCR** covers six visual scenarios and encompasses 150 meticulously designed questions categorized into six reasoning challenges. Additionally, Reasoning-OCR minimizes the impact of field-specialized knowledge. 
Our evaluation offers some insights for proprietary and open-source LMMs in different reasoning challenges, underscoring the urgent to improve the reasoning performance.
We hope Reasoning-OCR can inspire and facilitate future research on enhancing complex reasoning ability based on OCR cues.




# JSON Sample

```json
{
    "img": "1.png",
    "q_id": 1,
    "question": "What is the maximum value of the gap between the proportion of Democrats who do not hold favorable views of NATO and the proportion of Republicans who hold favorable views of NATO, when comparing these two proportions within the same year?",
    "question_c": "在同一年中，比较不持支持态度的民主党人比例与持支持态度的共和党人比例，二者之间差距的最大值是多少？",
    "answer": "28%",
    "hint": "Democrats who do not hold favorable views while Republicans who hold favorable views",
    "datasource": "chartQA-train-127.png",
    "scene":"chart",
    "type":"data comparison analysis"
},
```



## Field Explanations

| Field | Type | Description |
|------------------|----------|----------------------------------------------------------|
| "img" | string | The image name |
| "q_id" | int | The question index |
| "question" | string | The content of the question in English |
| "question_c" | string | The content of the question in Chinese (translated version) |
| "answer" | string | The concise answer to the question |
| "hint" | string | The hint provided to aid in answering the question |
| "datasource" | string | The source of the image |
| "scene" | string | The scene category associated with the image |
| "type" | string | The reasoning type of the question |



# Dataset Usage

For more evaluation details, please see our [GitHub repo](https://github.com/Hxyz-123/ReasoningOCR) for reference.



# License

Reasoning-OCR is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

The raw image in are collected from [ChartQA](https://github.com/vis-nlp/ChartQA), [DocQA](https://github.com/anisha2102/docvqa), [DT-VQA](https://github.com/ShuoZhang2003/DT-VQA), and websites.