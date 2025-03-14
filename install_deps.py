#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
依赖安装脚本
这个脚本帮助用户解决依赖冲突问题，分步骤安装所需的包
"""

import os
import sys
import subprocess
import platform

def run_command(command):
    """运行命令并打印输出"""
    print(f"执行: {command}")
    process = subprocess.Popen(
        command, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    for line in process.stdout:
        print(line.strip())
    
    process.wait()
    return process.returncode

def install_base_dependencies():
    """安装基础依赖"""
    print("=== 步骤1: 安装基础依赖 ===")
    
    # 安装核心包
    core_packages = [
        "python-dotenv>=1.0.0",
        "pandas>=2.1.1",
        "requests>=2.31.0"
    ]
    
    cmd = f"{sys.executable} -m pip install {' '.join(core_packages)}"
    return run_command(cmd) == 0

def install_langchain_dependencies():
    """安装LangChain相关依赖"""
    print("\n=== 步骤2: 安装LangChain相关依赖 ===")
    
    # 先安装langchain-core
    core_cmd = f"{sys.executable} -m pip install langchain-core>=0.3.41,<0.4.0"
    if run_command(core_cmd) != 0:
        return False
    
    # 安装其他langchain包
    langchain_packages = [
        "langchain-text-splitters>=0.3.6,<1.0.0",
        "langchain-community>=0.3.0,<0.4.0",
        "langchain-openai>=0.3.0,<0.4.0",
        "langchain>=0.3.20,<0.4.0"
    ]
    
    for package in langchain_packages:
        cmd = f"{sys.executable} -m pip install {package}"
        if run_command(cmd) != 0:
            return False
    
    return True

def install_model_dependencies():
    """安装模型相关依赖"""
    print("\n=== 步骤3: 安装模型相关依赖 ===")
    
    model_packages = [
        "openai>=1.58.1",
        "tiktoken>=0.7",
        "langchain-huggingface>=0.0.3",
        "langchain-openrouter>=0.0.1"
    ]
    
    cmd = f"{sys.executable} -m pip install {' '.join(model_packages)}"
    return run_command(cmd) == 0

def install_document_dependencies():
    """安装文档处理相关依赖"""
    print("\n=== 步骤4: 安装文档处理相关依赖 ===")
    
    doc_packages = [
        "pdfminer.six>=20221105",
        "python-docx>=0.8.11",
        "markdown>=3.4.4",
        "faiss-cpu>=1.7.4",
        "chromadb>=0.4.22"
    ]
    
    cmd = f"{sys.executable} -m pip install {' '.join(doc_packages)}"
    if run_command(cmd) != 0:
        return False
    
    # 安装可能会有冲突的包
    print("\n安装额外文档处理依赖...")
    extra_packages = [
        "pymupdf>=1.21.1",
        "unstructured>=0.10.30",
        "nltk>=3.8.1"
    ]
    
    for package in extra_packages:
        cmd = f"{sys.executable} -m pip install {package}"
        run_command(cmd)  # 即使失败也继续
    
    return True

def install_ml_dependencies():
    """安装机器学习相关依赖"""
    print("\n=== 步骤5: 安装机器学习相关依赖 ===")
    
    ml_packages = [
        "sentence-transformers>=2.2.2",
        "transformers>=4.35.0",
        "accelerate>=0.20.0"
    ]
    
    cmd = f"{sys.executable} -m pip install {' '.join(ml_packages)}"
    if run_command(cmd) != 0:
        return False
    
    # 单独安装torch，因为它很大
    print("\n安装PyTorch...")
    torch_cmd = f"{sys.executable} -m pip install torch>=2.0.0"
    run_command(torch_cmd)  # 即使失败也继续
    
    return True

def download_nltk_data():
    """下载NLTK数据"""
    print("\n=== 步骤6: 下载NLTK数据 ===")
    
    try:
        import nltk
        nltk.download('punkt')
        print("NLTK数据下载成功")
        return True
    except Exception as e:
        print(f"NLTK数据下载失败: {e}")
        return False

def main():
    """主函数"""
    print("=== 文档聊天系统依赖安装脚本 ===")
    print(f"Python版本: {platform.python_version()}")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print("开始安装依赖...\n")
    
    steps = [
        ("基础依赖", install_base_dependencies),
        ("LangChain依赖", install_langchain_dependencies),
        ("模型依赖", install_model_dependencies),
        ("文档处理依赖", install_document_dependencies),
        ("机器学习依赖", install_ml_dependencies),
        ("NLTK数据", download_nltk_data)
    ]
    
    success_count = 0
    for name, func in steps:
        if func():
            success_count += 1
            print(f"\n✓ {name}安装成功")
        else:
            print(f"\n✗ {name}安装过程中出现问题")
    
    print("\n=== 安装完成 ===")
    print(f"成功完成 {success_count}/{len(steps)} 个步骤")
    
    if success_count == len(steps):
        print("\n所有依赖安装成功！现在可以运行文档聊天系统了。")
    else:
        print("\n部分依赖安装可能有问题。但程序仍可能正常运行，因为我们提供了备选方案。")
        print("如果遇到问题，请参考README.md中的故障排除部分。")
    
    print("\n运行方式: python document_chat.py")

if __name__ == "__main__":
    main() 