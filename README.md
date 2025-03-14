# 文档聊天系统

这是一个基于LangChain 0.3版本的文档聊天系统，允许用户导入各种格式的文件，并通过与大模型聊天的方式针对导入的文件内容进行交互。此外，系统还支持灵活配置各种AI模型，包括文本生成、嵌入、图像生成和视频生成模型。

## 功能特点

- 支持多种文件格式：PDF、Word、Excel、CSV、Markdown、TXT等
- 使用向量数据库进行语义搜索
- 保存聊天历史，支持上下文理解
- 基于检索增强生成(RAG)技术提高回答准确性
- 灵活配置各种AI模型，包括：
  - 文本生成模型（OpenAI、OpenRouter、HuggingFace等）
  - 嵌入模型（OpenAI、HuggingFace、硅基流动等）
  - 图像生成模型（DALL-E、Stable Diffusion等）
  - 视频生成模型（Sora等）
- 在对话中直接切换不同模型
- 内置帮助系统，随时查看可用命令

## 安装

1. 克隆仓库：

```bash
git clone <repository-url>
cd <repository-directory>
```

2. 安装依赖：

### 使用安装脚本（推荐）

我们提供了一个安装脚本，可以帮助您解决依赖冲突问题：

```bash
python install_deps.py
```

这个脚本会分步骤安装所有必要的依赖，并处理可能的冲突。

### 手动安装

#### 使用conda创建虚拟环境
```bash
conda create -n langchain-0.3 python=3.9
conda activate langchain-0.3
```

#### 使用pip安装依赖
```bash
pip install -r requirements.txt
```

#### 使用国内镜像加速安装
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 安装额外依赖
某些文件格式需要额外的依赖：

```bash
# 对于Windows用户，可能需要先安装Visual C++ Build Tools
# 下载地址：https://visualstudio.microsoft.com/visual-cpp-build-tools/

# 对于Markdown和Excel文件
pip install "unstructured[all-docs]"
python -m nltk.downloader punkt

# 对于PDF文件
pip install pymupdf

# 对于HuggingFace模型
pip install transformers torch accelerate
```

如果安装过程中遇到问题，系统会自动使用备选的加载器，但功能可能会受到一定限制。

### 注意事项
如果安装过程中遇到依赖冲突，特别是关于包版本的问题，请注意以下要求：
- LangChain 0.3版本需要较新版本的OpenAI包（>=1.58.1）
- langchain-openai 0.3.0需要较新版本的tiktoken（>=0.7）

如果仍然遇到依赖冲突，可以尝试使用以下命令安装，让pip自动解决依赖问题：
```bash
pip install -r requirements.txt --no-deps
pip install langchain>=0.3.20,<0.4.0 langchain-community>=0.3.0,<0.4.0 langchain-openai>=0.3.0,<0.4.0 langchain-core>=0.3.41,<0.4.0
```

或者使用我们提供的安装脚本：
```bash
python install_deps.py
```

3. 配置环境变量：

复制`.env.example`文件并重命名为`.env`：
```bash
cp .env.example .env
```

然后编辑`.env`文件，填入你的API密钥和其他配置。系统支持灵活配置各种模型：

```
# 当前使用的模型配置
CURRENT_TEXT_MODEL=TEXT_MODEL_1
CURRENT_EMBEDDING_MODEL=EMBEDDING_MODEL_1
CURRENT_IMAGE_MODEL=IMAGE_MODEL_1
CURRENT_VIDEO_MODEL=VIDEO_MODEL_1

# 文本生成模型配置
TEXT_MODEL_1_NAME=gpt-3.5-turbo
TEXT_MODEL_1_API_BASE=https://api.openai.com/v1
TEXT_MODEL_1_API_KEY=your-api-key-here
TEXT_MODEL_1_TYPE=openai  # 可选：openai, openrouter, huggingface

# ModelScope API示例
TEXT_MODEL_2_NAME=qwen-max
TEXT_MODEL_2_API_BASE=https://api-inference.modelscope.cn/v1
TEXT_MODEL_2_API_KEY=your-api-key-here
TEXT_MODEL_2_TYPE=openai

# 本地HuggingFace模型示例
TEXT_MODEL_3_NAME=mistralai/Mistral-7B-Instruct-v0.2
TEXT_MODEL_3_TYPE=huggingface
TEXT_MODEL_3_LOCAL=true

# 嵌入模型配置
EMBEDDING_MODEL_1_NAME=text-embedding-ada-002
EMBEDDING_MODEL_1_API_BASE=https://api.openai.com/v1
EMBEDDING_MODEL_1_API_KEY=your-api-key-here
EMBEDDING_MODEL_1_TYPE=openai

# 硅基流动API示例
EMBEDDING_MODEL_2_NAME=BAAI/bge-large-zh-v1.5
EMBEDDING_MODEL_2_API_BASE=https://api.siliconflow.cn/v1
EMBEDDING_MODEL_2_API_KEY=your-api-key-here
EMBEDDING_MODEL_2_TYPE=openai
```

**重要提示**：
- 必须设置`CURRENT_TEXT_MODEL`和`CURRENT_EMBEDDING_MODEL`以及相应的API密钥，否则程序将无法运行。图像和视频模型是可选的。
- 对于硅基流动API，请确保API基础URL设置为`https://api.siliconflow.cn/v1`，系统会自动处理嵌入请求。

## 使用方法

运行主程序：

```bash
python document_chat.py
```

系统启动后，您可以：
1. 直接开始聊天 - 系统会使用纯文本模型回答您的问题
2. 输入"加载文件"或"新文件"命令 - 系统会提示您输入文件路径
3. 加载文件后，系统会自动切换到文档问答模式，使用检索增强生成(RAG)技术回答问题

### 特殊命令

系统支持多种特殊命令，可以通过输入"帮助"或"指令"随时查看所有可用命令：

- **帮助命令**：
  - `帮助` 或 `指令` - 显示所有可用命令和功能

- **文件操作**：
  - `加载文件` 或 `新文件` - 加载新的文件
  - 支持的文件格式：PDF、Word、Excel、CSV、Markdown、TXT

- **聊天功能**：
  - 直接输入问题即可与系统对话
  - 如果已加载文件，系统会基于文件内容回答
  - 如果未加载文件，系统会进行普通对话
  - `清除历史` - 清除当前聊天历史

- **模型管理**：
  - `列出模型` - 显示所有可用的模型配置
  - `切换文本模型:编号` - 切换到指定的文本模型
  - `切换嵌入模型:编号` - 切换到指定的嵌入模型
  - `切换图像模型:编号` - 切换到指定的图像模型
  - `切换视频模型:编号` - 切换到指定的视频模型

- **媒体生成**：
  - `生成图像:提示词` - 使用配置的图像模型生成图像
  - `生成视频:提示词` - 使用配置的视频模型生成视频
  - `查看图像` - 在浏览器中查看所有生成的图像和视频

- **其他功能**：
  - `退出` - 退出程序

### 模型切换功能

系统支持在对话中直接切换不同的模型，无需重启程序：

1. **查看可用模型**：输入`列出模型`命令，系统会显示所有配置的模型及其当前状态。

2. **切换模型**：使用`切换[类型]模型:[编号]`格式切换模型，例如：
   - `切换文本模型:2` - 切换到第二个文本模型
   - `切换嵌入模型:1` - 切换到第一个嵌入模型
   - `切换图像模型:3` - 切换到第三个图像模型
   - `切换视频模型:2` - 切换到第二个视频模型

3. **特殊处理**：
   - 切换嵌入模型时，如果已加载文档，系统会自动重新处理文档
   - 系统会检查模型是否存在，避免切换到不存在的模型

### 图像和视频生成功能

系统支持通过简单的命令生成图像和视频：

1. **生成图像**：输入`生成图像:你的提示词`，系统会调用配置的图像模型生成图像。
   例如：`生成图像:一只在草地上奔跑的金毛犬`

2. **生成视频**：输入`生成视频:你的提示词`，系统会调用配置的视频模型生成视频。
   例如：`生成视频:一只猫在追逐蝴蝶`

3. **查看生成的媒体**：输入`查看图像`命令，系统会自动打开浏览器，显示所有生成的图像和视频。
   - 网页界面会展示图像缩略图和视频链接
   - 每个图像和视频都会显示生成时使用的提示词和生成时间
   - 点击"刷新页面"按钮可以查看最新生成的内容

系统会在启动时自动启动一个本地Web服务器（默认端口2000），用于展示生成的媒体内容。
如果您的系统不支持自动打开浏览器，可以手动访问`http://localhost:2000`查看内容。

## 支持的文件格式

- PDF (.pdf)
- Word文档 (.docx, .doc)
- Excel表格 (.xlsx, .xls)
- CSV文件 (.csv)
- Markdown文件 (.md)
- 文本文件 (.txt)

## 技术架构

- 使用LangChain 0.3框架
- 支持多种AI模型：
  - 文本生成：OpenAI、OpenRouter、HuggingFace等
  - 嵌入：OpenAI、HuggingFace、硅基流动等
  - 图像生成：DALL-E、Stable Diffusion等
  - 视频生成：Sora等
- 使用FAISS向量数据库进行相似度搜索
- 使用RecursiveCharacterTextSplitter进行文本分割

## 故障排除

如果遇到以下错误：
- `openai.OpenAIError: The api_key client option must be set...`：请检查你的`.env`文件，确保已正确设置API密钥。
- `unstructured package not found`：请安装unstructured包，参考"安装额外依赖"部分。
- `PyMuPDF package not found`：请安装pymupdf包，参考"安装额外依赖"部分。
- `Error code: 404`：API端点不存在或无法访问，请检查API配置。如果使用硅基流动API，确保只设置基本域名（如`https://api.siliconflow.cn`）。
- 依赖冲突：请使用`python install_deps.py`脚本安装依赖，或参考"注意事项"部分的说明。
- 文件加载失败：确保文件路径正确，并且文件格式受支持。
- 图像生成失败：如果使用本地图像服务器，请确保服务器正在运行并且配置正确。系统现在支持多种响应格式，并提供备选的请求方法。

### 常见依赖问题解决方案

1. **依赖冲突**：使用我们提供的安装脚本`python install_deps.py`，它会分步骤安装依赖，避免冲突。

2. **无法安装某些包**：系统设计了备选方案，即使某些包安装失败，程序仍可能正常运行。例如：
   - 如果`pymupdf`安装失败，系统会自动使用`pdfminer.six`作为备选PDF加载器
   - 如果`unstructured`安装失败，系统会使用简单的文本加载器

3. **OpenAI API错误**：确保在`.env`文件中正确设置了API密钥和API基础URL。如果使用的是非官方API，请确保URL格式正确。

4. **ModelScope API错误**：
   - 系统会自动检测ModelScope API并启用流式模式
   - 如果仍然遇到问题，请确保API密钥正确
   - ModelScope API的基础URL应设置为`https://api-inference.modelscope.cn/v1`

5. **硅基流动API错误**：
   - 确保API基础URL设置为`https://api.siliconflow.cn/v1`
   - 不要添加额外的路径，系统会自动处理嵌入请求
   - 如果仍然遇到问题，请检查API密钥是否正确
   - 系统现在使用自定义嵌入模型类处理硅基流动API请求，支持批处理和错误恢复
   - 硅基流动API可能对文本长度和请求频率有限制，系统会自动进行批处理以避免这些问题
   - 对于大型文档，系统会自动使用更小的块大小（500字符）和批处理大小（4个文本/批次）
   - 文本会自动截断至1000字符以避免请求过大错误
   - 即使部分嵌入请求失败，系统也会继续处理，确保整个流程不会中断

6. **HuggingFace模型问题**：对于本地HuggingFace模型，确保安装了`transformers`、`torch`和`accelerate`包。

### 本地图像服务器配置

系统支持使用本地图像服务器（如运行在本地的Stable Diffusion服务器）。配置方法如下：

1. 在`.env`文件中设置本地服务器信息：
   ```
   IMAGE_MODEL_1_NAME=your-model-name
   IMAGE_MODEL_1_API_BASE=http://localhost:8004
   IMAGE_MODEL_1_API_KEY=your-api-key-if-needed
   IMAGE_MODEL_1_TYPE=openai
   ```

2. 确保本地服务器支持OpenAI兼容的API格式。系统会尝试以下方法调用本地服务器：
   - 首先使用requests直接发送HTTP请求
   - 如果失败，会尝试使用OpenAI客户端

3. 如果遇到错误，请检查日志获取详细信息。系统会记录请求和响应的详细信息，帮助诊断问题。

4. 本地服务器的响应格式应该类似于：
   ```json
   {
     "data": [
       {
         "url": "http://localhost:8004/output/image.png"
       }
     ]
   }
   ```

5. 如果本地服务器使用不同的响应格式，系统会尝试多种方式解析响应，以获取图像URL。

## 将项目上传至GitHub

如果您想将此项目上传到GitHub，请按照以下步骤操作：

### 1. 创建GitHub账号和仓库

1. 如果您还没有GitHub账号，请先在[GitHub](https://github.com)注册一个账号。
2. 登录后，点击右上角的"+"图标，选择"New repository"。
3. 填写仓库名称（如"document-chat-system"）和描述。
4. 选择是否将仓库设为公开或私有。
5. 可以选择初始化仓库时添加README文件、.gitignore文件和许可证。
6. 点击"Create repository"创建仓库。

### 2. 配置Git

如果您的电脑上还没有配置Git，请先安装Git并进行基本配置：

```bash
# 设置用户名和邮箱
git config --global user.name "您的GitHub用户名"
git config --global user.email "您的邮箱地址"

# 可选：设置默认编辑器
git config --global core.editor "您喜欢的编辑器"
```

### 3. 初始化本地仓库并上传代码

如果您的项目还没有初始化为Git仓库，请在项目目录下执行以下命令：

```bash
# 初始化Git仓库
git init

# 添加.gitignore文件，避免上传敏感信息和不必要的文件
echo ".env" > .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
echo ".DS_Store" >> .gitignore
echo "venv/" >> .gitignore
echo ".idea/" >> .gitignore
echo ".vscode/" >> .gitignore

# 添加所有文件到暂存区
git add .

# 提交更改
git commit -m "初始提交：文档聊天图片视频系统"

# 添加远程仓库
git remote add origin https://github.com/您的用户名/您的仓库名.git

# 推送到GitHub
git push -u origin master  # 或 git push -u origin main（取决于您的默认分支名称）
```

### 4. 更新和维护

后续当您对代码进行更改后，可以使用以下命令更新GitHub上的代码：

```bash
# 查看更改状态
git status

# 添加更改的文件
git add .

# 提交更改
git commit -m "更新说明：例如添加了新功能或修复了bug"

# 推送到GitHub
git push
```

### 5. 注意事项

- **不要上传敏感信息**：确保`.env`文件已添加到`.gitignore`中，避免API密钥等敏感信息被上传。
- **添加详细的README**：确保README.md文件包含项目的详细说明、安装步骤和使用方法。
- **添加许可证**：考虑为您的项目添加适当的开源许可证。
- **版本控制**：使用语义化版本控制（Semantic Versioning）来管理您的项目版本。
- **分支管理**：对于重大功能开发或实验性功能，考虑使用分支进行开发。

## 贡献指南

欢迎对本项目进行贡献！您可以通过以下方式参与：

1. 提交Issue：报告bug或提出新功能建议
2. 提交Pull Request：修复bug或实现新功能
3. 改进文档：完善README或添加更详细的使用说明
4. 分享使用经验：在Issues中分享您使用本系统的经验和建议

## 许可证

本项目采用MIT许可证。详情请参阅LICENSE文件。
