import json
import os

def extract_key_info(input_file: str):
    """提取关键信息并保存到新文件"""
    # 读取原始文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取关键信息
    extracted_data = []
    for item in data:
        extracted_item = {
            "task": item["task"],
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "answer": item["answer"]
        }
        extracted_data.append(extracted_item)
    
    # 生成新的文件名
    base_name = os.path.splitext(input_file)[0]
    output_file = f"{base_name}_extracted.json"
    
    # 保存提取后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, indent=2, ensure_ascii=False)
    
    print(f"提取完成！结果已保存到: {output_file}")

def main():
    # 处理所有结果文件
    result_files = [
        "results/genehop_result.json",
        "results/geneturing_result.json",
        "results/geneturing-10Qs_result.json"
    ]
    
    for file in result_files:
        if os.path.exists(file):
            print(f"\n处理文件: {file}")
            extract_key_info(file)
        else:
            print(f"\n文件不存在: {file}")

def main():
    # 处理所有结果文件
    result_files = [
        "results/genehop_result.json",
        "results/geneturing_result.json",
    ]
    
    for file in result_files:
        if os.path.exists(file):
            print(f"\n处理文件: {file}")
            extract_key_info(file)
        else:
            print(f"\n文件不存在: {file}")

if __name__ == "__main__":
    main()