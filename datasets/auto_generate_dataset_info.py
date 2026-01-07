import re
import os, json, gc
import argparse
import os
import sys

from numpy import sort

def get_all_files(directory):
    """递归获取目录下所有文件路径"""
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            file_list.append(full_path)
    return sort(file_list)


def json_comment_parser(file_path):
    """
    读取含注释的 JSON 文件并转换为字典
    支持处理以下注释类型：
    - 单行注释 // comment
    - 块注释 /* comment */
    """
    # 定义注释正则表达式
    comment_pattern = r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"'
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = ''.join(f.readlines())
        
        # 分步处理注释
        def replace_comment(match):
            s = match.group(0)
            return '' if s.startswith(('//', '/*')) else s
        
        # 使用正则表达式过滤注释
        cleaned = re.sub(
            comment_pattern, 
            replace_comment, 
            content, 
            flags=re.MULTILINE|re.DOTALL
        )
        
        # 删除多余逗号（JSON 严格模式校验）
        cleaned = re.sub(r',\s*(?=[}\]])', '', cleaned)
        
        return json.loads(cleaned)

def extract_tfrecord_number(filename):
    """
    从文件名中提取 tfrecord 编号
    :param filename: 符合 bridge_dataset-train.tfrecord-00000-of-01024 格式的字符串
    :return: 提取的整型数字 (如 00000 → 0)
    """
    pattern = r"tfrecord-(\d+)-of"  # 捕获 tfrecord- 和 -of 之间的数字
    match = re.search(pattern, filename)
    
    if not match:
        raise ValueError(f"文件名格式不符合要求: {filename}")
    
    return int(match.group(1))  # 自动去除前导零

def extract_aviable_indexs(original_list, indexs_list):
    """
    从list中保留有效的index元素
    Args:
        origin_list (list): _description_
        indexs_list (list): _description_
    """
    sorted_indices = sorted(indexs_list)
    original_list[:] = [
        original_list[i] 
        for i in sorted_indices 
        if 0 <= i < len(original_list)
        and i not in sorted_indices[:sorted_indices.index(i)]  # 去重
    ]
    return original_list


def main():
    parser = argparse.ArgumentParser(description="处理命令行参数示例")
    parser.add_argument('--path', type=str, help='数据集文件夹路径')
    parser.add_argument('--info_file', type=str, help='dataset_info.json 文件路径')
    args = parser.parse_args()

    # 验证路径有效性
    if not os.path.exists(args.path):
        print(f"错误：路径 '{args.path}' 不存在", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(args.path):
        print(f"错误：'{args.path}' 不是目录", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.info_file):
        print(f"错误：'{args.info_file}' 配置文件不存在", file=sys.stderr)
        sys.exit(1)

    # 获取并输出文件列表
    train_file_list = []    # 训练集文件列表
    train_file_indexs = []
    val_file_list = []      # 测试集文件列表
    val_file_indexs = []
    try:
        files = get_all_files(args.path)
        for idx, file in enumerate(files, 1):
            if 'train' in file:
                train_file_list.append(file)
            elif 'val' in file:
                val_file_list.append(file)
            else:
                continue
    except Exception as e:
        print(f"扫描过程中发生错误：{str(e)}", file=sys.stderr)
        gc.collect()
        sys.exit(1)

    # 打印训练集与验证集
    print(f"训练集文件共 [{len(train_file_list)}], 如下：")
    for i in range(len(train_file_list)):
        filename = train_file_list[i]
        train_file_indexs.append(extract_tfrecord_number(filename))
        print(f"\t[{i+1}/{len(train_file_list)}]: {filename} | index={extract_tfrecord_number(filename)}")
    print(f"验证集文件共 [{len(val_file_list)}], 如下：")
    for i in range(len(val_file_list)):
        filename = val_file_list[i]
        val_file_indexs.append(extract_tfrecord_number(filename))
        print(f"\t[{i+1}/{len(val_file_list)}]: {filename} | index={extract_tfrecord_number(filename)}")
    
    # 读取 dataset_info.json 配置文件
    try:
        dataset_info = json_comment_parser(args.info_file)
        print(f"配置文件 {args.info_file} 加载成功")
    except Exception as e:
        print(f"错误：'{args.info_file}' 配置文件不加载失败", file=sys.stderr)
        sys.exit(1)

    extract_aviable_indexs(dataset_info["splits"][0]["shardLengths"], train_file_indexs)
    extract_aviable_indexs(dataset_info["splits"][1]["shardLengths"], val_file_indexs)

    # 写文件
    try:
        save_file_name = "./dataset_info_new.json"
        with open(save_file_name, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=4)  # 
        print(f"文件保存成功 {save_file_name}")
    except Exception as e:
        print(f"写本地文件失败 {e}")

if __name__ == "__main__":
    print(11111)
    main()
