import os
import sys
import re
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dotenv import load_dotenv
import traceback

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

# 尝试导入必要的包
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import (
        PyMuPDFLoader, PDFMinerLoader, CSVLoader, UnstructuredExcelLoader,
        Docx2txtLoader, TextLoader, UnstructuredMarkdownLoader
    )
    from langchain_community.embeddings import (
        HuggingFaceEmbeddings,
        OpenAIEmbeddings,
    )
    from langchain_openai import ChatOpenAI
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.chat_models import ChatOllama
    from langchain_huggingface import HuggingFaceEndpoint
    
    # 尝试导入图像和视频生成相关的包
    try:
        from langchain_openai import OpenAI
        from openai import OpenAI as DirectOpenAI
        HAS_OPENAI = True
    except ImportError:
        HAS_OPENAI = False
        logger.warning("OpenAI package not found. Image and video generation may not work.")
    
    try:
        from PIL import Image
        import io
        import base64
        import requests
        HAS_IMAGE_LIBS = True
    except ImportError:
        HAS_IMAGE_LIBS = False
        logger.warning("PIL, io, base64, or requests not found. Image display may not work.")
    
    # 尝试导入网页服务器相关的包
    try:
        import threading
        import webbrowser
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import socketserver
        import json
        import urllib.parse
        HAS_WEB_SERVER = True
    except ImportError:
        HAS_WEB_SERVER = False
        logger.warning("Web server packages not found. Web display may not work.")
        
    HAS_LANGCHAIN = True
except ImportError as e:
    HAS_LANGCHAIN = False
    logger.error(f"Error importing LangChain packages: {e}")
    print(f"Error importing LangChain packages: {e}")
    print("Please install the required packages: pip install -r requirements.txt")
    sys.exit(1)

# 全局变量，用于存储生成的图像URL
generated_images = []
generated_videos = []

# 简单的HTTP服务器，用于显示生成的图像和视频
class ImageDisplayHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>文档聊天系统 - 图像和视频展示</title>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        max-width: 1200px;
                        margin: 0 auto;
                        padding: 20px;
                        background-color: #f5f5f5;
                    }
                    h1 {
                        color: #333;
                        text-align: center;
                    }
                    .gallery {
                        display: grid;
                        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                        gap: 20px;
                        margin-top: 20px;
                    }
                    .image-card {
                        background: white;
                        border-radius: 8px;
                        overflow: hidden;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                        transition: transform 0.3s ease;
                    }
                    .image-card:hover {
                        transform: translateY(-5px);
                    }
                    .image-card img {
                        width: 100%;
                        height: auto;
                        display: block;
                    }
                    .image-info {
                        padding: 15px;
                    }
                    .video-card {
                        background: white;
                        border-radius: 8px;
                        overflow: hidden;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                        padding: 15px;
                        margin-bottom: 20px;
                    }
                    .video-card a {
                        color: #0066cc;
                        text-decoration: none;
                        font-weight: bold;
                    }
                    .video-card a:hover {
                        text-decoration: underline;
                    }
                    .refresh {
                        display: block;
                        margin: 20px auto;
                        padding: 10px 20px;
                        background-color: #4CAF50;
                        color: white;
                        border: none;
                        border-radius: 4px;
                        cursor: pointer;
                        font-size: 16px;
                    }
                    .refresh:hover {
                        background-color: #45a049;
                    }
                    .empty-message {
                        text-align: center;
                        margin: 50px 0;
                        color: #666;
                    }
                </style>
            </head>
            <body>
                <h1>文档聊天系统 - 图像和视频展示</h1>
                <button class="refresh" onclick="location.reload()">刷新页面</button>
                
                <h2>生成的图像</h2>
                <div class="gallery" id="images-container">
                    <!-- 图像将在这里动态加载 -->
                </div>
                
                <h2>生成的视频</h2>
                <div id="videos-container">
                    <!-- 视频将在这里动态加载 -->
                </div>
                
                <script>
                    // 获取图像数据
                    fetch('/api/images')
                        .then(response => response.json())
                        .then(data => {
                            const container = document.getElementById('images-container');
                            if (data.length === 0) {
                                container.innerHTML = '<p class="empty-message">暂无生成的图像</p>';
                                return;
                            }
                            
                            data.forEach((item, index) => {
                                const card = document.createElement('div');
                                card.className = 'image-card';
                                card.innerHTML = `
                                    <img src="${item.url}" alt="生成的图像 ${index + 1}">
                                    <div class="image-info">
                                        <p><strong>提示词:</strong> ${item.prompt}</p>
                                        <p><small>生成时间: ${item.timestamp}</small></p>
                                    </div>
                                `;
                                container.appendChild(card);
                            });
                        });
                    
                    // 获取视频数据
                    fetch('/api/videos')
                        .then(response => response.json())
                        .then(data => {
                            const container = document.getElementById('videos-container');
                            if (data.length === 0) {
                                container.innerHTML = '<p class="empty-message">暂无生成的视频</p>';
                                return;
                            }
                            
                            data.forEach((item, index) => {
                                const card = document.createElement('div');
                                card.className = 'video-card';
                                card.innerHTML = `
                                    <h3>视频 ${index + 1}</h3>
                                    <p><strong>提示词:</strong> ${item.prompt}</p>
                                    <p><a href="${item.url}" target="_blank">点击查看视频</a></p>
                                    <p><small>生成时间: ${item.timestamp}</small></p>
                                `;
                                container.appendChild(card);
                            });
                        });
                </script>
            </body>
            </html>
            """
            
            self.wfile.write(html.encode())
        
        elif self.path == '/api/images':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(generated_images).encode())
        
        elif self.path == '/api/videos':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(generated_videos).encode())
        
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')

class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    pass

def start_web_server(port=8000):
    """启动Web服务器"""
    server = ThreadedHTTPServer(('localhost', port), ImageDisplayHandler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    logger.info(f"Web服务器已启动，访问 http://localhost:{port} 查看生成的图像和视频")
    return server

# 自定义Markdown加载器
class SimpleMarkdownLoader:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def load(self):
        from langchain_core.documents import Document
        with open(self.file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        metadata = {"source": self.file_path}
        return [Document(page_content=content, metadata=metadata)]

def get_model_config(model_type: str, model_key: str = None) -> Dict[str, Any]:
    """
    获取指定类型和键的模型配置
    
    Args:
        model_type: 模型类型，如 'TEXT', 'EMBEDDING', 'IMAGE', 'VIDEO'
        model_key: 模型键，如 'TEXT_MODEL_1'，如果为None则使用当前配置
        
    Returns:
        包含模型配置的字典
    """
    if model_key is None:
        # 获取当前配置的模型键
        current_key_env = f"CURRENT_{model_type}_MODEL"
        model_key = os.getenv(current_key_env)
        if not model_key:
            raise ValueError(f"环境变量 {current_key_env} 未设置")
    
    # 获取模型配置
    model_name = os.getenv(f"{model_key}_NAME")
    model_api_base = os.getenv(f"{model_key}_API_BASE")
    model_api_key = os.getenv(f"{model_key}_API_KEY")
    model_type_value = os.getenv(f"{model_key}_TYPE", "openai").lower()
    model_local = os.getenv(f"{model_key}_LOCAL", "false").lower() == "true"
    
    if not model_name:
        raise ValueError(f"模型名称未设置: {model_key}_NAME")
    
    return {
        "name": model_name,
        "api_base": model_api_base,
        "api_key": model_api_key,
        "type": model_type_value,
        "local": model_local,
        "key": model_key
    }

def init_text_model():
    """初始化文本生成模型"""
    try:
        config = get_model_config("TEXT")
        logger.info(f"初始化文本模型: {config['name']} (类型: {config['type']})")
        
        if config['type'] == 'openai':
            if not config['api_key']:
                raise ValueError("OpenAI API密钥未设置")
            
            # 检查是否是ModelScope API
            is_modelscope = config['api_base'] and "modelscope.cn" in config['api_base']
            
            # ModelScope API需要启用流式模式
            streaming = True if is_modelscope else False
            
            if is_modelscope:
                logger.info("检测到ModelScope API，启用流式模式")
            
            return ChatOpenAI(
                model=config['name'],
                openai_api_key=config['api_key'],
                openai_api_base=config['api_base'] if config['api_base'] else None,
                temperature=0.7,
                streaming=streaming
            )
        elif config['type'] == 'huggingface':
            if config['local']:
                # 本地HuggingFace模型
                try:
                    from langchain_community.llms import HuggingFacePipeline
                    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
                    
                    model_id = config['name']
                    tokenizer = AutoTokenizer.from_pretrained(model_id)
                    model = AutoModelForCausalLM.from_pretrained(model_id)
                    
                    pipe = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=512
                    )
                    
                    return HuggingFacePipeline(pipeline=pipe)
                except Exception as e:
                    logger.error(f"初始化本地HuggingFace模型失败: {e}")
                    raise
            else:
                # 远程HuggingFace模型
                if not config['api_key']:
                    raise ValueError("HuggingFace API密钥未设置")
                
                return HuggingFaceEndpoint(
                    endpoint_url=f"https://api-inference.huggingface.co/models/{config['name']}",
                    huggingfacehub_api_token=config['api_key'],
                    task="text-generation",
                    model_kwargs={"temperature": 0.7, "max_new_tokens": 512}
                )
        elif config['type'] == 'openrouter':
            if not config['api_key']:
                raise ValueError("OpenRouter API密钥未设置")
            
            from langchain_openrouter import ChatOpenRouter
            return ChatOpenRouter(
                model=config['name'],
                api_key=config['api_key'],
                temperature=0.7
            )
        elif config['type'] == 'ollama':
            # 本地Ollama模型
            return ChatOllama(model=config['name'])
        else:
            raise ValueError(f"不支持的模型类型: {config['type']}")
    
    except Exception as e:
        logger.error(f"初始化文本模型失败: {e}")
        logger.error(traceback.format_exc())
        
        # 尝试使用备用模型
        try:
            logger.info("尝试使用备用文本模型: gpt-3.5-turbo")
            return ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7
            )
        except Exception as backup_error:
            logger.error(f"初始化备用文本模型失败: {backup_error}")
            raise ValueError("无法初始化任何文本模型，请检查配置和API密钥")

def init_embedding():
    """初始化嵌入模型"""
    try:
        config = get_model_config("EMBEDDING")
        logger.info(f"初始化嵌入模型: {config['name']} (类型: {config['type']})")
        
        if config['type'] == 'openai':
            if not config['api_key']:
                raise ValueError("OpenAI API密钥未设置")
            
            # 检查是否是硅基流动API
            api_base = config['api_base'] if config['api_base'] else None
            is_siliconflow = api_base and "siliconflow.cn" in api_base
            
            if is_siliconflow:
                # 完全重写硅基流动API的URL路径，确保正确
                # 提取基本域名部分
                if "://" in api_base:
                    protocol, rest = api_base.split("://", 1)
                    domain = rest.split("/")[0]
                    api_base = f"{protocol}://{domain}/v1"
                else:
                    domain = api_base.split("/")[0]
                    api_base = f"https://{domain}/v1"
                
                logger.info(f"检测到硅基流动API，设置URL为: {api_base}")
                
                # 为硅基流动API创建自定义嵌入模型
                from langchain_core.embeddings import Embeddings
                
                class SiliconFlowEmbeddings(Embeddings):
                    def __init__(self, api_key, api_base, model):
                        self.api_key = api_key
                        self.api_base = api_base
                        self.model = model
                        self.client = None
                        self._init_client()
                    
                    def _init_client(self):
                        import httpx
                        self.client = httpx.Client(
                            base_url=self.api_base,
                            headers={"Authorization": f"Bearer {self.api_key}"}
                        )
                    
                    def _truncate_text(self, text, max_length=1000):
                        """截断文本以避免请求过大"""
                        if len(text) > max_length:
                            logger.warning(f"文本长度 {len(text)} 超过限制，截断至 {max_length} 字符")
                            return text[:max_length]
                        return text
                    
                    def embed_documents(self, texts):
                        """嵌入文档列表"""
                        # 硅基流动API对文本长度有限制，这里进行分批处理
                        batch_size = 4  # 减小批处理大小，避免请求过大
                        embeddings = []
                        
                        for i in range(0, len(texts), batch_size):
                            batch_texts = texts[i:i+batch_size]
                            # 截断每个文本，避免请求过大
                            batch_texts = [self._truncate_text(text) for text in batch_texts]
                            
                            try:
                                response = self.client.post(
                                    "/embeddings",
                                    json={
                                        "model": self.model,
                                        "input": batch_texts,
                                        "encoding_format": "float"
                                    }
                                )
                                response.raise_for_status()
                                data = response.json()
                                
                                # 提取嵌入向量
                                batch_embeddings = [item["embedding"] for item in data["data"]]
                                embeddings.extend(batch_embeddings)
                                logger.info(f"成功嵌入批次 {i//batch_size + 1}，共 {len(batch_texts)} 个文本")
                            except Exception as e:
                                logger.error(f"嵌入文档批次 {i} 失败: {e}")
                                # 如果失败，为每个文本添加空向量
                                empty_embedding = [0.0] * 1024  # 假设向量维度为1024
                                embeddings.extend([empty_embedding] * len(batch_texts))
                        
                        return embeddings
                    
                    def embed_query(self, text):
                        """嵌入单个查询"""
                        # 截断查询文本，避免请求过大
                        text = self._truncate_text(text)
                        
                        try:
                            response = self.client.post(
                                "/embeddings",
                                json={
                                    "model": self.model,
                                    "input": text,
                                    "encoding_format": "float"
                                }
                            )
                            response.raise_for_status()
                            data = response.json()
                            
                            # 提取嵌入向量
                            return data["data"][0]["embedding"]
                        except Exception as e:
                            logger.error(f"嵌入查询失败: {e}")
                            # 返回空向量
                            return [0.0] * 1024  # 假设向量维度为1024
                
                return SiliconFlowEmbeddings(
                    api_key=config['api_key'],
                    api_base=api_base,
                    model=config['name']
                )
            else:
                # 标准OpenAI嵌入
                return OpenAIEmbeddings(
                    model=config['name'],
                    openai_api_key=config['api_key'],
                    openai_api_base=api_base
                )
        elif config['type'] == 'huggingface':
            # HuggingFace嵌入模型
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': True}
            
            return HuggingFaceEmbeddings(
                model_name=config['name'],
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
        else:
            raise ValueError(f"不支持的嵌入模型类型: {config['type']}")
    
    except Exception as e:
        logger.error(f"初始化嵌入模型失败: {e}")
        logger.error(traceback.format_exc())
        
        # 尝试使用备用嵌入模型
        try:
            logger.info("尝试使用备用嵌入模型: sentence-transformers/all-MiniLM-L6-v2")
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': True}
            
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
        except Exception as backup_error:
            logger.error(f"初始化备用嵌入模型失败: {backup_error}")
            raise ValueError("无法初始化任何嵌入模型，请检查配置和API密钥")

def init_image_model():
    """初始化图像生成模型"""
    if not HAS_OPENAI:
        logger.warning("OpenAI包未安装，无法使用图像生成功能")
        return None
    
    try:
        config = get_model_config("IMAGE")
        logger.info(f"初始化图像模型: {config['name']} (类型: {config['type']})")
        
        if config['type'] == 'openai':
            if not config['api_key']:
                raise ValueError("OpenAI API密钥未设置")
            
            client = DirectOpenAI(
                api_key=config['api_key'],
                base_url=config['api_base'] if config['api_base'] else "https://api.openai.com/v1"
            )
            return {
                "client": client,
                "model": config['name'],
                "type": "openai"
            }
        else:
            raise ValueError(f"不支持的图像模型类型: {config['type']}")
    
    except Exception as e:
        logger.error(f"初始化图像模型失败: {e}")
        logger.error(traceback.format_exc())
        return None

def init_video_model():
    """初始化视频生成模型"""
    if not HAS_OPENAI:
        logger.warning("OpenAI包未安装，无法使用视频生成功能")
        return None
    
    try:
        config = get_model_config("VIDEO")
        logger.info(f"初始化视频模型: {config['name']} (类型: {config['type']})")
        
        if config['type'] == 'openai':
            if not config['api_key']:
                raise ValueError("OpenAI API密钥未设置")
            
            client = DirectOpenAI(
                api_key=config['api_key'],
                base_url=config['api_base'] if config['api_base'] else "https://api.openai.com/v1"
            )
            return {
                "client": client,
                "model": config['name'],
                "type": "openai"
            }
        else:
            raise ValueError(f"不支持的视频模型类型: {config['type']}")
    
    except Exception as e:
        logger.error(f"初始化视频模型失败: {e}")
        logger.error(traceback.format_exc())
        return None

def generate_image(image_model, prompt: str, size: str = "1024x1024") -> Optional[str]:
    """
    生成图像
    
    Args:
        image_model: 图像模型
        prompt: 提示词
        size: 图像尺寸
        
    Returns:
        图像URL或None
    """
    if not image_model:
        return None
    
    try:
        if image_model["type"] == "openai":
            # 检查是否是本地服务器（如localhost）
            is_local_server = False
            api_base = image_model["client"].base_url
            
            # 将URL对象转换为字符串进行检查
            api_base_str = str(api_base) if api_base else ""
            
            # 检查是否是ModelScope API
            is_modelscope = "modelscope.cn" in api_base_str
            if is_modelscope:
                logger.info(f"检测到ModelScope API: {api_base_str}")
                try:
                    # ModelScope API使用正确的端点
                    import requests
                    import json
                    
                    # 构建正确的API URL - 使用官方端点
                    if "://" in api_base_str:
                        protocol, rest = api_base_str.split("://", 1)
                        domain = rest.split("/")[0]
                        api_url = f"{protocol}://{domain}/v1/images/generations"
                    else:
                        api_url = f"https://{api_base_str}/v1/images/generations"
                    
                    logger.info(f"使用ModelScope API URL: {api_url}")
                    
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {image_model['client'].api_key}"
                    }
                    
                    # 构建请求数据
                    data = {
                        "model": image_model["model"],
                        "prompt": prompt,
                        "n": 1,
                        "size": size,
                        "response_format": "url"
                    }
                    
                    # 发送请求
                    logger.info(f"发送请求到ModelScope: {api_url}")
                    logger.info(f"请求数据: {data}")
                    response = requests.post(api_url, headers=headers, json=data)
                    response.raise_for_status()
                    
                    # 解析响应
                    result = response.json()
                    logger.info(f"ModelScope响应: {result}")
                    
                    # 从响应中提取URL
                    if 'data' in result and len(result['data']) > 0 and 'url' in result['data'][0]:
                        url = result['data'][0]['url']
                        logger.info(f"从ModelScope响应中提取的URL: {url}")
                    # 检查ModelScope特定的响应格式
                    elif 'images' in result and len(result['images']) > 0 and 'url' in result['images'][0]:
                        url = result['images'][0]['url']
                        logger.info(f"从ModelScope images字段提取的URL: {url}")
                    else:
                        # 如果没有URL，尝试从其他字段提取
                        logger.info("标准URL字段不存在，尝试从其他字段提取")
                        
                        # 检查是否包含图像数据或其他URL字段
                        if 'data' in result and len(result['data']) > 0:
                            data_item = result['data'][0]
                            if 'b64_json' in data_item:
                                # 处理Base64编码的图像
                                from PIL import Image
                                import io
                                import base64
                                import tempfile
                                import os
                                
                                base64_data = data_item['b64_json']
                                image_data = base64.b64decode(base64_data)
                                
                                # 保存到临时文件
                                temp_dir = tempfile.gettempdir()
                                temp_file = os.path.join(temp_dir, f"generated_image_{hash(prompt)}.png")
                                
                                with open(temp_file, "wb") as f:
                                    f.write(image_data)
                                
                                # 使用文件URL
                                url = f"file://{temp_file}"
                                logger.info(f"创建了Base64图像文件: {url}")
                            elif 'image_path' in data_item:
                                url = data_item['image_path']
                                logger.info(f"从image_path字段提取的URL: {url}")
                            else:
                                # 如果没有找到URL，创建一个本地占位图像
                                logger.info("响应中没有找到URL，创建占位图像")
                                from PIL import Image, ImageDraw, ImageFont
                                import tempfile
                                import os
                                
                                # 创建一个简单的图像，显示提示词
                                img = Image.new('RGB', (800, 600), color=(73, 109, 137))
                                d = ImageDraw.Draw(img)
                                
                                # 尝试加载字体，如果失败则使用默认字体
                                try:
                                    font = ImageFont.truetype("arial.ttf", 20)
                                except IOError:
                                    font = ImageFont.load_default()
                                
                                # 在图像上绘制文本
                                d.text((10, 10), f"提示词: {prompt}", fill=(255, 255, 0), font=font)
                                d.text((10, 50), "ModelScope未返回图像URL", fill=(255, 255, 0), font=font)
                                d.text((10, 90), f"响应: {str(result)[:200]}...", fill=(255, 255, 0), font=font)
                                
                                # 保存到临时文件
                                temp_dir = tempfile.gettempdir()
                                temp_file = os.path.join(temp_dir, f"generated_image_{hash(prompt)}.png")
                                img.save(temp_file)
                                
                                # 使用文件URL
                                url = f"file://{temp_file}"
                                logger.info(f"创建了本地占位图像: {url}")
                        else:
                            logger.error(f"无法从ModelScope响应中提取URL: {result}")
                            return None
                    
                    # 添加到生成的图像列表
                    from datetime import datetime
                    generated_images.append({
                        "url": url,
                        "prompt": prompt,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    return url
                
                except Exception as modelscope_error:
                    logger.error(f"使用ModelScope生成图像失败: {modelscope_error}")
                    logger.error(traceback.format_exc())
                    
                    # 尝试使用ModelScope的另一个端点
                    try:
                        logger.info("尝试使用ModelScope的另一个端点")
                        
                        import requests
                        import json
                        
                        # 构建另一个可能的API URL - 尝试第三个已知的端点
                        if "://" in api_base_str:
                            protocol, rest = api_base_str.split("://", 1)
                            domain = rest.split("/")[0]
                            api_url = f"{protocol}://{domain}/api/v1/models/damo/cv_diffusion_text-to-image-synthesis/inference"
                        else:
                            api_url = f"https://{api_base_str}/api/v1/models/damo/cv_diffusion_text-to-image-synthesis/inference"
                        
                        logger.info(f"使用ModelScope备选API URL: {api_url}")
                        
                        headers = {
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {image_model['client'].api_key}"
                        }
                        
                        # 构建请求数据
                        data = {
                            "prompt": prompt,
                            "width": int(size.split("x")[0]),
                            "height": int(size.split("x")[1])
                        }
                        
                        # 发送请求
                        logger.info(f"发送请求到ModelScope备选端点: {api_url}")
                        logger.info(f"请求数据: {data}")
                        response = requests.post(api_url, headers=headers, json=data)
                        response.raise_for_status()
                        
                        # 解析响应
                        result = response.json()
                        logger.info(f"ModelScope备选端点响应: {result}")
                        
                        # 从响应中提取URL或图像数据
                        if 'output' in result and 'output_imgs' in result['output'] and len(result['output']['output_imgs']) > 0:
                            # 处理Base64编码的图像
                            from PIL import Image
                            import io
                            import base64
                            import tempfile
                            import os
                            
                            base64_data = result['output']['output_imgs'][0]
                            image_data = base64.b64decode(base64_data)
                            
                            # 保存到临时文件
                            temp_dir = tempfile.gettempdir()
                            temp_file = os.path.join(temp_dir, f"generated_image_{hash(prompt)}.png")
                            
                            with open(temp_file, "wb") as f:
                                f.write(image_data)
                            
                            # 使用文件URL
                            url = f"file://{temp_file}"
                            logger.info(f"创建了Base64图像文件: {url}")
                            
                            # 添加到生成的图像列表
                            from datetime import datetime
                            generated_images.append({
                                "url": url,
                                "prompt": prompt,
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                            
                            return url
                        else:
                            logger.error(f"无法从ModelScope备选端点响应中提取图像数据: {result}")
                            return None
                    
                    except Exception as backup_error:
                        logger.error(f"使用ModelScope备选端点生成图像失败: {backup_error}")
                        logger.error(traceback.format_exc())
                        return None
            
            # 检查是否是OpenRouter API
            is_openrouter = "openrouter.ai" in api_base_str
            if is_openrouter:
                logger.info(f"检测到OpenRouter API: {api_base_str}")
                try:
                    # OpenRouter不支持标准的images.generate方法，需要使用chat.completions
                    logger.info("使用OpenRouter的聊天API生成图像")
                    
                    # 使用requests直接调用API
                    import requests
                    import json
                    
                    # 构建API URL
                    api_url = f"{api_base_str}/chat/completions"
                    if not api_url.startswith("http"):
                        api_url = f"https://{api_url}"
                    
                    # 修复URL中可能出现的双斜杠问题
                    api_url = api_url.replace("//chat", "/chat")
                    
                    logger.info(f"使用OpenRouter API URL: {api_url}")
                    
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {image_model['client'].api_key}"
                    }
                    
                    # 检查是否是Stable Diffusion模型
                    is_stable_diffusion = "stable-diffusion" in image_model["model"].lower()
                    
                    # 构建请求数据
                    if is_stable_diffusion:
                        # 对于Stable Diffusion模型，使用特定的提示词格式
                        logger.info(f"检测到Stable Diffusion模型: {image_model['model']}，使用特定提示词格式")
                        data = {
                            "model": image_model["model"],
                            "messages": [
                                {"role": "system", "content": "你是一个图像生成助手。请使用Stable Diffusion生成高质量图像。"},
                                {"role": "user", "content": f"使用Stable Diffusion生成图像：{prompt}。请直接返回图像URL，不要有任何文字说明。"}
                            ],
                            "temperature": 0.7,
                            "max_tokens": 1024
                        }
                    else:
                        # 构建请求数据，使用通用提示词格式
                        data = {
                            "model": image_model["model"],
                            "messages": [
                                {"role": "system", "content": "你是一个图像生成助手。请根据用户的提示生成高质量的图像。"},
                                {"role": "user", "content": f"生成一张图片：{prompt}。请直接返回图像，不要有任何文字说明。"}
                            ],
                            "temperature": 0.7,
                            "max_tokens": 1024
                        }
                        
                        # 检查是否是支持图像生成的模型
                        if "gemma" in image_model["model"].lower() or "claude" in image_model["model"].lower():
                            # 这些模型可能不支持图像生成，尝试使用特定的图像生成模型
                            logger.info(f"检测到可能不支持图像生成的模型: {image_model['model']}，尝试使用备选模型")
                            
                            # 尝试使用支持图像生成的模型
                            image_models = [
                                "stability.stable-diffusion-xl-1.0:free",
                                "anthropic.claude-3-haiku",
                                "openai/dall-e-3",
                                "midjourney"
                            ]
                            
                            # 使用第一个可用的图像模型
                            data["model"] = image_models[0]
                            logger.info(f"切换到图像生成模型: {data['model']}")
                            
                            # 修改提示词格式，使其更适合图像生成
                            data["messages"] = [
                                {"role": "system", "content": "You are an image generation assistant. Generate high-quality images based on user prompts."},
                                {"role": "user", "content": f"Generate an image of: {prompt}"}
                            ]
                    
                    # 发送请求
                    logger.info(f"发送请求到OpenRouter: {api_url}")
                    logger.info(f"请求数据: {data}")
                    response = requests.post(api_url, headers=headers, json=data)
                    response.raise_for_status()
                    
                    # 解析响应
                    result = response.json()
                    logger.info(f"OpenRouter响应: {result}")
                    
                    # 从响应中提取文本
                    if 'choices' in result and len(result['choices']) > 0 and 'message' in result['choices'][0]:
                        generated_text = result['choices'][0]['message']['content']
                        logger.info(f"OpenRouter生成的文本: {generated_text}")
                        
                        # 检查是否包含URL或Base64图像
                        import re
                        url_match = re.search(r'https?://\S+', generated_text)
                        base64_match = re.search(r'data:image/\w+;base64,([^"\'\\s]+)', generated_text)
                        
                        if url_match:
                            url = url_match.group(0)
                            # 清理URL，移除可能的引号或括号
                            url = url.rstrip(',.;:!?)]}\'\"')
                            logger.info(f"从响应中提取的URL: {url}")
                        elif base64_match:
                            # 处理Base64编码的图像
                            from PIL import Image
                            import io
                            import base64
                            import tempfile
                            import os
                            
                            base64_data = base64_match.group(1)
                            image_data = base64.b64decode(base64_data)
                            
                            # 保存到临时文件
                            temp_dir = tempfile.gettempdir()
                            temp_file = os.path.join(temp_dir, f"generated_image_{hash(prompt)}.png")
                            
                            with open(temp_file, "wb") as f:
                                f.write(image_data)
                            
                            # 使用文件URL
                            url = f"file://{temp_file}"
                            logger.info(f"创建了Base64图像文件: {url}")
                        else:
                            # 如果没有URL或Base64，创建一个本地占位图像
                            logger.info("响应中没有URL或Base64图像，创建占位图像")
                            from PIL import Image, ImageDraw, ImageFont
                            import tempfile
                            import os
                            
                            # 创建一个简单的图像，显示提示词
                            img = Image.new('RGB', (800, 600), color=(73, 109, 137))
                            d = ImageDraw.Draw(img)
                            
                            # 尝试加载字体，如果失败则使用默认字体
                            try:
                                font = ImageFont.truetype("arial.ttf", 20)
                            except IOError:
                                font = ImageFont.load_default()
                            
                            # 在图像上绘制文本
                            d.text((10, 10), f"提示词: {prompt}", fill=(255, 255, 0), font=font)
                            d.text((10, 50), "OpenRouter未返回图像URL", fill=(255, 255, 0), font=font)
                            d.text((10, 90), f"生成的文本: {generated_text[:200]}...", fill=(255, 255, 0), font=font)
                            
                            # 保存到临时文件
                            temp_dir = tempfile.gettempdir()
                            temp_file = os.path.join(temp_dir, f"generated_image_{hash(prompt)}.png")
                            img.save(temp_file)
                            
                            # 使用文件URL
                            url = f"file://{temp_file}"
                            logger.info(f"创建了本地占位图像: {url}")
                    else:
                        logger.error(f"无法从OpenRouter响应中提取文本: {result}")
                        return None
                    
                    # 添加到生成的图像列表
                    from datetime import datetime
                    generated_images.append({
                        "url": url,
                        "prompt": prompt,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    return url
                
                except Exception as openrouter_error:
                    logger.error(f"使用OpenRouter生成图像失败: {openrouter_error}")
                    logger.error(traceback.format_exc())
                    return None
            
            # 标准OpenAI API
            try:
                response = image_model["client"].images.generate(
                    model=image_model["model"],
                    prompt=prompt,
                    size=size,
                    quality="standard",
                    n=1,
                )
                url = response.data[0].url
                
                # 添加到生成的图像列表
                from datetime import datetime
                generated_images.append({
                    "url": url,
                    "prompt": prompt,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                return url
            except Exception as api_error:
                logger.error(f"使用标准API生成图像失败: {api_error}")
                logger.error(traceback.format_exc())
                return None
        else:
            raise ValueError(f"不支持的图像模型类型: {image_model['type']}")
    except Exception as e:
        logger.error(f"生成图像失败: {e}")
        logger.error(traceback.format_exc())
        return None

def display_image(url: str) -> None:
    """
    显示图像
    
    Args:
        url: 图像URL
    """
    if not HAS_IMAGE_LIBS:
        print(f"图像URL: {url}")
        return
    
    try:
        response = requests.get(url)
        image = Image.open(io.BytesIO(response.content))
        
        # 在终端中显示图像URL
        print(f"图像URL: {url}")
        
        # 如果在支持图像显示的环境中，可以取消下面的注释
        # image.show()
    except Exception as e:
        logger.error(f"显示图像失败: {e}")
        print(f"图像URL: {url}")

def open_web_browser(url: str) -> None:
    """
    打开Web浏览器
    
    Args:
        url: 要打开的URL
    """
    try:
        webbrowser.open(url)
    except Exception as e:
        logger.error(f"打开Web浏览器失败: {e}")
        print(f"请手动访问: {url}")

class DocumentProcessor:
    def __init__(self):
        """初始化文档处理器"""
        self.text_model = init_text_model()
        self.embedding_model = init_embedding()
        self.image_model = init_image_model()
        self.video_model = init_video_model()
        self.vectorstore = None
        self.chat_history = []
        self.current_file = None
        
        # 文件加载器映射
        self.loaders = {
            ".pdf": self._load_pdf,
            ".docx": self._load_docx,
            ".xlsx": self._load_excel,
            ".xls": self._load_excel,
            ".csv": self._load_csv,
            ".md": self._load_markdown,
            ".txt": self._load_text
        }
    
    def _load_pdf(self, file_path: str):
        """加载PDF文件"""
        try:
            return PyMuPDFLoader(file_path).load()
        except ImportError:
            print("PyMuPDF包未找到，尝试使用PDFMinerLoader...")
            return PDFMinerLoader(file_path).load()
    
    def _load_docx(self, file_path: str):
        """加载Word文档"""
        return Docx2txtLoader(file_path).load()
    
    def _load_excel(self, file_path: str):
        """加载Excel文件"""
        try:
            return UnstructuredExcelLoader(file_path).load()
        except ImportError:
            print("Unstructured包未找到，尝试使用pandas加载Excel...")
            # 使用pandas加载Excel
            df = pd.read_excel(file_path)
            from langchain_core.documents import Document
            return [Document(page_content=df.to_string(), metadata={"source": file_path})]
    
    def _load_csv(self, file_path: str):
        """加载CSV文件"""
        return CSVLoader(file_path).load()
    
    def _load_markdown(self, file_path: str):
        """加载Markdown文件"""
        try:
            return UnstructuredMarkdownLoader(file_path).load()
        except ImportError:
            print("Unstructured包未找到，尝试使用简单Markdown加载器...")
            return SimpleMarkdownLoader(file_path).load()
    
    def _load_text(self, file_path: str):
        """加载文本文件"""
        return TextLoader(file_path, encoding="utf-8").load()
    
    def load_document(self, file_path: str):
        """
        加载文档
        
        Args:
            file_path: 文件路径
            
        Returns:
            加载的文档
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext not in self.loaders:
            raise ValueError(f"不支持的文件类型: {file_ext}")
        
        print(f"正在加载文件: {file_path}")
        self.current_file = file_path
        
        try:
            documents = self.loaders[file_ext](file_path)
            print(f"文档加载成功，共 {len(documents)} 个片段")
            return documents
        except Exception as e:
            logger.error(f"加载文档失败: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def process_document(self, documents):
        """
        处理文档
        
        Args:
            documents: 文档列表
            
        Returns:
            处理后的向量存储
        """
        print("正在处理文档...")
        
        # 检查是否使用硅基流动API
        is_siliconflow = False
        if hasattr(self.embedding_model, 'api_base') and hasattr(self.embedding_model, 'model'):
            is_siliconflow = "siliconflow.cn" in getattr(self.embedding_model, 'api_base', '')
        
        # 分割文档，对硅基流动API使用更小的块大小
        chunk_size = 500 if is_siliconflow else 1000
        chunk_overlap = 100 if is_siliconflow else 200
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        splits = text_splitter.split_documents(documents)
        print(f"文档分割完成，共 {len(splits)} 个块")
        
        if is_siliconflow:
            print(f"检测到硅基流动API，使用较小的块大小 ({chunk_size} 字符)和批处理")
        
        # 创建向量存储
        self.vectorstore = FAISS.from_documents(splits, self.embedding_model)
        print("向量存储创建完成")
        
        return self.vectorstore
    
    def get_retriever(self):
        """获取检索器"""
        if not self.vectorstore:
            raise ValueError("向量存储未初始化，请先加载并处理文档")
        
        return self.vectorstore.as_retriever(search_kwargs={"k": 5})
    
    def format_chat_history(self):
        """格式化聊天历史"""
        formatted_history = []
        
        for message in self.chat_history:
            if isinstance(message, HumanMessage):
                formatted_history.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                formatted_history.append(f"AI: {message.content}")
        
        return "\n".join(formatted_history)
    
    def generate_response(self, query: str):
        """
        生成回复
        
        Args:
            query: 用户查询
            
        Returns:
            生成的回复
        """
        # 检查是否是特殊命令
        if query.lower() in ["帮助", "指令", "help", "commands"]:
            return """文档聊天系统支持以下功能：

1. 文件操作：
   - '加载文件' 或 '新文件' - 加载新文件
   - 支持的文件格式：PDF、Word、Excel、CSV、Markdown、TXT

2. 聊天功能：
   - 直接输入问题即可与系统对话
   - 如果已加载文件，系统会基于文件内容回答
   - 如果未加载文件，系统会进行普通对话

3. 图像和视频生成：
   - '生成图像:提示词' - 生成图像
   - '生成视频:提示词' - 生成视频
   - '查看图像' - 在浏览器中查看生成的图像和视频

4. 模型管理：
   - '列出模型' - 显示所有可用的模型配置
   - '切换文本模型:编号' - 切换到指定的文本模型
   - '切换嵌入模型:编号' - 切换到指定的嵌入模型
   - '切换图像模型:编号' - 切换到指定的图像模型
   - '切换视频模型:编号' - 切换到指定的视频模型

5. 其他功能：
   - '清除历史' - 清除聊天历史
   - '退出' - 退出程序
   - '帮助' 或 '指令' - 显示此帮助信息

您可以根据需要随时使用这些命令。如果您想开始使用，可以：
1. 直接开始聊天
2. 输入'加载文件'来加载文档
3. 使用'生成图像:提示词'来生成图像
4. 使用'列出模型'查看可用的模型配置"""
        
        if query.lower() == "新文件" or query.lower() == "加载文件":
            self.vectorstore = None
            self.chat_history = []
            return "已清除当前文件，请输入文件路径加载新文件。"
        
        if query.lower() == "清除历史":
            self.chat_history = []
            return "已清除聊天历史。"
        
        if query.lower() == "退出":
            return "再见！"
        
        # 检查是否是切换模型命令
        model_switch_match = re.match(r"^切换(文本|嵌入|图像|视频)模型[:：](\d+)$", query)
        if model_switch_match:
            model_type = model_switch_match.group(1)
            model_number = model_switch_match.group(2)
            
            # 映射中文模型类型到英文
            type_mapping = {
                "文本": "TEXT",
                "嵌入": "EMBEDDING",
                "图像": "IMAGE",
                "视频": "VIDEO"
            }
            
            if model_type in type_mapping:
                model_type_eng = type_mapping[model_type]
                model_key = f"{model_type_eng}_MODEL_{model_number}"
                
                # 检查模型是否存在
                model_name = os.getenv(f"{model_key}_NAME")
                if not model_name:
                    return f"模型 {model_key} 不存在，请检查配置。"
                
                # 更新环境变量
                os.environ[f"CURRENT_{model_type_eng}_MODEL"] = model_key
                
                # 重新初始化相应的模型
                if model_type == "文本":
                    self.text_model = init_text_model()
                elif model_type == "嵌入":
                    self.embedding_model = init_embedding()
                    # 如果已加载文档，需要重新处理
                    if self.vectorstore and self.current_file:
                        try:
                            documents = self.load_document(self.current_file)
                            self.process_document(documents)
                            return f"已切换到{model_type}模型 {model_number}，并重新处理当前文档。"
                        except Exception as e:
                            return f"已切换到{model_type}模型 {model_number}，但重新处理文档时出错: {e}"
                elif model_type == "图像":
                    self.image_model = init_image_model()
                elif model_type == "视频":
                    self.video_model = init_video_model()
                
                return f"已切换到{model_type}模型 {model_number}。"
            else:
                return "不支持的模型类型，请使用'文本'、'嵌入'、'图像'或'视频'。"
        
        # 检查是否是列出模型命令
        if query.lower() == "列出模型":
            response = "当前可用的模型配置:\n\n"
            
            # 获取所有环境变量
            env_vars = os.environ
            
            # 查找所有模型配置
            model_types = ["TEXT", "EMBEDDING", "IMAGE", "VIDEO"]
            for model_type in model_types:
                response += f"【{model_type}模型】\n"
                current_model = os.getenv(f"CURRENT_{model_type}_MODEL", "未设置")
                response += f"当前使用: {current_model}\n"
                
                # 查找该类型的所有模型
                for i in range(1, 10):  # 假设最多有9个模型配置
                    model_key = f"{model_type}_MODEL_{i}"
                    model_name = os.getenv(f"{model_key}_NAME")
                    if model_name:
                        model_api_base = os.getenv(f"{model_key}_API_BASE", "默认")
                        model_type_value = os.getenv(f"{model_key}_TYPE", "openai")
                        response += f"{model_key}: {model_name} (类型: {model_type_value}, API: {model_api_base})\n"
                
                response += "\n"
            
            response += "使用'切换[类型]模型:[编号]'命令切换模型，例如'切换文本模型:2'。"
            return response
        
        # 检查是否是图像生成命令
        image_match = re.match(r"^生成(图像|图片)[:：](.+)$", query)
        if image_match and self.image_model:
            prompt = image_match.group(2).strip()
            logger.info(f"正在生成图像，提示词: {prompt}")
            image_url = generate_image(self.image_model, prompt)
            
            if image_url:
                display_image(image_url)
                response = f"图像已生成: {image_url}\n您可以输入'查看图像'命令在浏览器中查看"
            else:
                response = "图像生成失败，请检查日志获取详细信息。"
            
            # 添加到聊天历史
            self.chat_history.append(HumanMessage(content=query))
            self.chat_history.append(AIMessage(content=response))
            
            return response
        
        # 检查是否是视频生成命令
        video_match = re.match(r"^生成视频[:：](.+)$", query)
        if video_match and self.video_model:
            prompt = video_match.group(1).strip()
            logger.info(f"正在生成视频，提示词: {prompt}")
            video_url = generate_video(self.video_model, prompt)
            
            if video_url:
                response = f"视频已生成: {video_url}\n您可以输入'查看图像'命令在浏览器中查看"
            else:
                response = "视频生成失败，请检查日志获取详细信息。"
            
            # 添加到聊天历史
            self.chat_history.append(HumanMessage(content=query))
            self.chat_history.append(AIMessage(content=response))
            
            return response
        
        # 如果没有加载文档，但也不是特殊命令，则使用纯文本模型回答
        if not self.vectorstore:
            # 构建简单的聊天提示模板
            template = """
            你是一个友好的AI助手。请回答用户的问题。
            如果用户想要加载文档，请告诉他们输入"加载文件"或"新文件"命令。
            
            聊天历史:
            {chat_history}
            
            用户问题: {question}
            
            请提供详细、准确的回答:
            """
            
            prompt = ChatPromptTemplate.from_template(template)
            
            # 构建简单的聊天链
            chat_chain = (
                {"question": RunnablePassthrough(), "chat_history": lambda _: self.format_chat_history()}
                | prompt
                | self.text_model
                | StrOutputParser()
            )
            
            # 生成回复
            response = chat_chain.invoke(query)
            
            # 添加到聊天历史
            self.chat_history.append(HumanMessage(content=query))
            self.chat_history.append(AIMessage(content=response))
            
            return response
        
        # 如果已加载文档，则使用RAG回答
        # 构建提示模板
        template = """
        你是一个专业的文档助手。根据以下检索到的文档内容和聊天历史，回答用户的问题。
        如果你不知道答案，请诚实地说你不知道，不要编造信息。
        
        聊天历史:
        {chat_history}
        
        检索到的文档内容:
        {context}
        
        用户问题: {question}
        
        请提供详细、准确的回答:
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # 构建检索增强生成链
        retriever = self.get_retriever()
        
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough(), "chat_history": lambda _: self.format_chat_history()}
            | prompt
            | self.text_model
            | StrOutputParser()
        )
        
        # 生成回复
        response = rag_chain.invoke(query)
        
        # 添加到聊天历史
        self.chat_history.append(HumanMessage(content=query))
        self.chat_history.append(AIMessage(content=response))
        
        return response

def main():
    """主函数"""
    print("欢迎使用文档聊天系统！")
    print("基于LangChain 0.3版本")
    print("支持的文件格式: PDF, Word, Excel, CSV, Markdown, TXT")
    print("输入'帮助'或'指令'可以查看所有可用功能")
    print()
    print("您可以直接开始聊天，或者输入'加载文件'命令导入文档进行问答。")
    print()
    
    # 启动Web服务器
    web_server_port = 2000
    if HAS_WEB_SERVER:
        try:
            server = start_web_server(web_server_port)
            print(f"Web服务器已启动，输入'查看图像'命令在浏览器中查看生成的图像和视频")
        except Exception as e:
            logger.error(f"启动Web服务器失败: {e}")
            print(f"启动Web服务器失败: {e}")
    else:
        print("Web服务器功能不可用，请安装必要的包")
    
    processor = DocumentProcessor()
    
    while True:
        if processor.vectorstore:
            status_msg = f"当前已加载文件: {processor.current_file}"
        else:
            status_msg = "当前未加载文件，您可以直接聊天或输入'加载文件'命令"
        
        query = input(f"\n{status_msg}\n请输入您的问题: ")
        
        if not query:
            continue
        
        if query.lower() == "退出":
            print("再见！")
            break
        
        if query.lower() == "查看图像":
            if HAS_WEB_SERVER:
                url = f"http://localhost:{web_server_port}"
                print(f"正在打开浏览器，查看生成的图像和视频: {url}")
                open_web_browser(url)
            else:
                print("Web服务器功能不可用，请安装必要的包")
            continue
        
        if query.lower() in ["加载文件", "新文件"]:
            processor.vectorstore = None
            processor.chat_history = []
            file_path = input("请输入文件路径: ")
            
            try:
                documents = processor.load_document(file_path)
                processor.process_document(documents)
                print(f"文件 '{file_path}' 已成功加载和处理")
            except Exception as e:
                print(f"处理文件时出错: {e}")
                continue
        else:
            try:
                # 检查是否是图像生成命令
                image_match = re.match(r"^生成(图像|图片)[:：](.+)$", query)
                if image_match and processor.image_model:
                    prompt = image_match.group(2).strip()
                    logger.info(f"正在生成图像，提示词: {prompt}")
                    image_url = generate_image(processor.image_model, prompt)
                    
                    if image_url:
                        display_image(image_url)
                        response = f"图像已生成: {image_url}\n您可以输入'查看图像'命令在浏览器中查看"
                    else:
                        response = "图像生成失败，请检查日志获取详细信息。"
                    
                    # 添加到聊天历史
                    processor.chat_history.append(HumanMessage(content=query))
                    processor.chat_history.append(AIMessage(content=response))
                    
                    print(f"\n{response}")
                    continue
                
                # 检查是否是视频生成命令
                video_match = re.match(r"^生成视频[:：](.+)$", query)
                if video_match and processor.video_model:
                    prompt = video_match.group(1).strip()
                    logger.info(f"正在生成视频，提示词: {prompt}")
                    video_url = generate_video(processor.video_model, prompt)
                    
                    if video_url:
                        response = f"视频已生成: {video_url}\n您可以输入'查看图像'命令在浏览器中查看"
                    else:
                        response = "视频生成失败，请检查日志获取详细信息。"
                    
                    # 添加到聊天历史
                    processor.chat_history.append(HumanMessage(content=query))
                    processor.chat_history.append(AIMessage(content=response))
                    
                    print(f"\n{response}")
                    continue
                
                response = processor.generate_response(query)
                print(f"\n{response}")
            except Exception as e:
                print(f"生成回复时出错: {e}")
                logger.error(f"生成回复时出错: {e}")
                logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 