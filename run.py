import ast
import matplotlib.pyplot as plt
import openai
import networkx as nx
import gradio as gr
import io, datetime
import numpy as np
import pandas as pd
from PIL import Image
import builtins
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False  
pre_prompt = """
# CONTEXT(上下文) 
我想根据用户的问题和提供的函数调用关系三元组,进行代码的影响分析
###
# OBJECTIVE(目标) 
帮我根据用户的问题和提供的函数调用关系三元组, 进行代码的影响分析

# ATTENTION(注意点) 
1、代码的分析要准确，不能出错
2、不能出现未出现过的函数名字
# STYLE(风格)
参照提供的代码分析例子来进行代码的影响分析
###
# TONE(语调) 
Professional, technical
###
# AUDIENCE(受众) 
代码的分析是要辅组开发工程师进行代码开发的，所以需要保证它的准确性
###
# RESPONSE(响应) 
保持代码分析的准确性
# Example
###
######
Example One:

问题: 如果我修改function1，会涉及到那些函数呢？

函数依赖关系:
[(function2, 在第11行调用, function3), (function1, 在第17行调用, function2)]

回答: 如果修改function1，可能会影响第17行的function2函数，而function2函数会在第11行调用function3函数，所以如果修改function1函数，可能会影响到function2函数和function3函数

######
# Attention(重要级第一)

1.不能出现未提供的函数依赖关系的其他函数信息
2.用户的问题没有问到具体的函数就不能进行函数分析，只能进行常识问答

###

# START Intent Recognition
######
用户的问题: {}

函数依赖关系: {}

函数的解释: {}

######
Answer:
"""
def is_builtin_function(name):
    return name in dir(builtins)
def extract_function_calls(file):
    '''提取代码的关系和注释'''
    if not isinstance(file, list):
            file = [file]
    functions = {}
    function_calls = []
    for i in range(len(file)):
        # filename = file[i].name
        filename = file[i]
        # 用来存储函数定义的字典
        # 用来存储函数调用的列表，每个元素是一个三元组 (调用的函数, 在哪一行被调用, 被调用的函数)
        # 解析文件，生成抽象语法树
        with open(filename, "r", encoding='utf=8') as file:
            tree = ast.parse(file.read(), filename=filename)
        # 遍历抽象语法树
        anotecate = get_function_comments(tree)
        for node in ast.walk(tree):
            # 检查是否是函数定义节点
            if isinstance(node, ast.FunctionDef):
                print(f"node.name******{node.name}")
                # 提取函数名和参数列表
                function_name = node.name
                arguments = [arg.arg for arg in node.args.args]
                functions[function_name] = arguments

            # 检查是否是函数调用节点
            elif isinstance(node, ast.Call):
                # 提取调用的函数名
                if isinstance(node.func, ast.Name):
                    function_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    function_name = node.func.attr
                if function_name and not is_builtin_function(function_name):
                    print(f"function_name******{function_name}")
                    # 提取所在的行号
                    line_number = node.lineno
                    # 记录调用关系
                    caller_function = find_caller_function(tree, line_number)
                    if caller_function:
                        function_calls.append((caller_function, f"在第{line_number}行调用", function_name))
        print(anotecate)
        return function_calls, anotecate
def get_function_comments(tree):
    '''获取函数注释'''
    function_comments = {}
    result = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # 获取函数名
            function_name = node.name
            print(f"function_name____________{function_name}")
            # 获取函数注释
            docstring = ast.get_docstring(node)
            print(f"******{docstring}")
            if docstring is not None:
                function_comments[function_name] = docstring
    print(function_comments)
    for i,j in function_comments.items():
        print(i)
        print(j)
        result.append((i,j))
    return result
def find_caller_function(tree, line_number):
    '''寻找调用的函数'''
    # 在抽象语法树中查找包含指定行号的函数
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.lineno < line_number and node.end_lineno > line_number:
                return node.name
    return None


def draw_function_calls_graph(function_calls):
    '''画函数关系图'''
    # 创建有向图对象
    G = nx.DiGraph()

    # 添加节点和边
    for caller, line_use, callee in function_calls:
        line_use = line_use.replace('在第','use func in line ').replace('行调用','')
        G.add_edge(caller, callee, name=line_use)

    # 绘制图形
         # draw graph with labels
    pos = nx.spring_layout(G)  # 定义布局
    fig = plt.figure(figsize=(10, 8))  # 设置图形大小
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color="lightblue", font_size=12, arrowsize=15)  # 绘制节点和边
    node_labels = nx.get_node_attributes(G, 'desc')
    nx.draw_networkx_labels(G, pos, labels=node_labels,font_size=7)
    edge_labels = nx.get_edge_attributes(G, 'name')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,font_size=7)
    plt.title("函数调用关系图")  # 设置标题
    plt.tight_layout()
    plt.savefig("function_calls_graph.png", format="png")  # 保存图形为PNG格式文件
    result = plot_to_numpy(fig)
    return result

def plot_to_numpy(figure):
    """
    Convert a Matplotlib figure to a numpy array representing an image.

    Args:
        figure (matplotlib.figure.Figure): Matplotlib figure to convert.

    Returns:
        numpy.array: Numpy array representing the image.
    """
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img)
    return img_array
def run_llm(prompt, history=[], functions=[], sys_prompt= f"The current time is {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S %A')}. You are an useful AI assistant that helps people solve the problem step by step."):

    openai.api_base = "http://localhost:8009/v1"
    openai.api_key = 'none'
    openai.api_request_timeout = 1 # 设置超时时间为10秒
    messages=[{"role": "system", "content": sys_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": "" + prompt})
    response = openai.ChatCompletion.create(
        model = "qwen_code",
        messages = messages,
        temperature=0.65,
        max_tokens=2048,
        functions=functions
        )
    data_res = response['choices'][0]['message']['content']
    function_call = response['choices'][0]['message']['function_call']
    return data_res, function_call
def model_chat(query, history, functions, function_comments):
    responses = list()
    function_image = draw_function_calls_graph(functions)
    if len(history) > 0:
        for history_msg in history:
            responses.append(history_msg)
    yield responses, function_image
    mylist = list()
    mylist.append(query)
    prompt = pre_prompt.format(query, str(functions), str(function_comments))
    answer, functions_call = run_llm(prompt=prompt)
    mylist.append(answer)
    
    
    responses.append(mylist)
    yield responses, function_image

def clear_session():
    return '', [], None

def main():
    with gr.Blocks(css="footer {visibility: hidden}",theme=gr.themes.Soft()) as demo:
        gr.Markdown("""<center><font size=10>Code analyze</center>""")
        with gr.Row(equal_height=False):
            file_output = gr.Files(height=200)
            output_list = gr.List()
            ouput_other_list = gr.List()
            
        with gr.Row(equal_height=False):
            gallery = gr.Image(label="代码依赖关系",scale=1)
            chatbot = gr.Chatbot(label='代码分析回答',scale=1)
        with gr.Row():
            textbox = gr.Textbox(lines=3, label='提出你想关于代码的问题')
        with gr.Row():
            with gr.Column():
                clear_history = gr.Button("🧹 clear")
                sumbit = gr.Button("🚀 submit")
        file_output.upload(extract_function_calls, inputs= [file_output],outputs=[output_list, ouput_other_list])
        sumbit.click(model_chat, [textbox, chatbot, output_list, ouput_other_list], [chatbot, gallery])
        clear_history.click(fn=clear_session,
                            inputs=[],
                            outputs=[textbox, chatbot, gallery])
    demo.queue(api_open=False).launch(server_name='0.0.0.0', server_port=7860,max_threads=10, height=800, share=False)
if __name__ == "__main__":
    main()
# 导出数据
# x, y = extract_function_calls('example.py')
# df1 = pd.DataFrame(x, columns=['source','link_value','target'])
# df2 = pd.DataFrame(y, columns=['functions','description'])
# df1.to_csv('links.csv',index=False)
# df2.to_csv('function_value.csv',index=False)
# print(df1)
# print(df2)