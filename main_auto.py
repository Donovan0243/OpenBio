import json
import os
import logging
from src.core.settings import configure_settings
from src.core.rag import initialize_rag_system
from langchain_core.messages import HumanMessage

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_questions(file_path: str) -> dict:
    """从JSON文件加载嵌套结构问题"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_results(results: list, output_file: str):
    """保存结果到JSON文件"""
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"结果已保存到: {output_file}")

def main():
    configure_settings()
    workflow = initialize_rag_system()
    
    # 指定输入和输出文件
    input_file = "data/geneturing.json"
    output_file = "results/geneturing_result.json"
    
    # 加载嵌套结构
    qas = load_questions(input_file)
    results = []
    
    # 检查是否有历史结果，支持断点续跑
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        done_set = {(r['task'], r['question']) for r in results}
    else:
        done_set = set()
    
    print(f"\n开始处理 {input_file}")
    print("="*50)
    
    for task, info in qas.items():
        for question, ground_truth in info.items():
            if (task, question) in done_set:
                print(f"跳过已处理: [{task}] {question}")
                continue
            
            print(f"\n任务类型: {task}")
            print(f"问题: {question}")
            print(f"标准答案: {ground_truth}")
            
            try:
                # 创建输入
                inputs = {
                    "messages": [
                        HumanMessage(content=question, additional_kwargs={"type": "user_question"})
                    ]
                }
                
                # 执行工作流并收集所有输出
                node_outputs = {}
                for output in workflow.stream(inputs):
                    node_name = list(output.keys())[0]
                    node_output = output[node_name]
                    
                    # 如果是消息列表，只取最后一条消息的内容
                    if isinstance(node_output, dict) and "messages" in node_output:
                        latest_message = node_output["messages"][-1].content
                        node_outputs[node_name] = latest_message
                    else:
                        node_outputs[node_name] = node_output
                
                # 获取最终答案（假设是最后一个节点的输出）
                final_answer = list(node_outputs.values())[-1] if node_outputs else "No answer generated"
                
                # 保存结果
                result = {
                    "task": task,
                    "question": question,
                    "ground_truth": ground_truth,
                    "answer": final_answer,
                    "node_outputs": node_outputs
                }
                results.append(result)
                
                print(f"答案: {final_answer}")
                print("="*50)
                
                # 每处理完一个问题就保存一次结果
                save_results(results, output_file)
                
            except Exception as e:
                logger.error(f"处理问题时出错: {str(e)}")
                # 保存错误信息
                result = {
                    "task": task,
                    "question": question,
                    "ground_truth": ground_truth,
                    "answer": f"Error: {str(e)}",
                    "node_outputs": {}
                }
                results.append(result)
                save_results(results, output_file)
                print(f"处理问题时出错: {str(e)}")
                print("="*50)
                continue

if __name__ == "__main__":
    main()