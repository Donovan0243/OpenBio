from src.core.settings import configure_settings
from src.core.rag import initialize_rag_system
from langchain_core.messages import HumanMessage, SystemMessage
from src.tools.get_prompt_header import get_prompt_header

def main():
    configure_settings()
    workflow = initialize_rag_system()
    
    # 获取长提示词
    # system_prompt = get_prompt_header([1, 1, 1, 1, 1, 1])
    # 准备问题列表
    questions = [
        # "What is the official gene symbol of SNAT6?", #SLC38A6
        # "What are genes related to Distal renal tubular acidosis?", #SLC4A1, ATP6V0A4
        # "Which chromosome is TTTY7 gene located on human genome?", #chrY
        "Align the DNA sequence to the human genome:GGACAGCTGAGATCACATCAAGGATTCCAGAAAGAATTGGCACAGGATCATTCAAGATGCATCTCTCCGTTGCCCCTGTTCCTGGCTTTCCTTCAACTTCCTCAAAGGGGACATCATTTCGGAGTTTGGCTTCCA", #chr8:7081648-7081782
        "Which organism does the DNA sequence come from:CGTACACCATTGGTGCCAGTGACTGTGGTCAATTCGGTAGAAGTAGAGGTAAAAGTGCTGTTCCATGGCTCAGTTGTAGTTATGATGGTGCTAGCAGTTGTTGGAGTTCTGATGACAATGACGGTTTCGTCAGTTG", #yeast
        "Convert ENSG00000205403 to official gene/ symbol.", #CFI
        "Is LOC124907753 a protein-coding gene?", #N/A
        "Which gene is SNP rs1241371358 associated with?", #LRRC23
        "Which chromosome does SNP rs545148486 locate on human genome?", #chr16
        "What is the function of the gene associated with SNP rs1241371358? Let's decompose the question to sub-questions and solve them step by step.", # Predicted to be active in cytosol.
        "List chromosome locations of the genes related to Hemolytic anemia due to phosphofructokinase deficiency. Let's decompose the question to sub-questions and solve them step by step.", #"21q22.3"
        "What are the aliases of the gene that contains this sequnece:ATTGTGAGAGTAACCAACGTGGGGTTACGGGGGAGAATCTGGAGAGAAGAGAAGAGGTTAACAACCCTCCCACTTCCTGGCCACCCCCCTCCACCTTTTCTGGTAAGGAGCCC. Let's decompose the question to sub-questions and solve them step by step."
    ]
    # 逐个处理问题
    for question in questions:
        print("\n" + "="*50)
        print(f"问题: {question}")
        
        # 创建输入 - 添加系统提示词作为历史的第一条消息
        inputs = {
            "messages": [
                # SystemMessage(content=system_prompt, additional_kwargs={"type": "system_prompt"}),  # 系统提示词作为第一条消息
                HumanMessage(content=question, additional_kwargs={"type": "user_question"})
            ]
        }
        
        # 执行工作流并获取输出
        for output in workflow.stream(inputs):
            node_name = list(output.keys())[0]
            node_output = output[node_name]
            
            print("\n" + "="*50)
            print(f"节点: {node_name}")
            
            # 如果是消息列表，只取最后一条消息的内容
            if isinstance(node_output, dict) and "messages" in node_output:
                latest_message = node_output["messages"][-1].content
                print(f"输出: {latest_message}")
            else:
                print(f"输出: {node_output}")
            print("="*50)

if __name__ == "__main__":
    main()