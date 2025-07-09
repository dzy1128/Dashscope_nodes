import os
from openai import OpenAI
from .get_image_base64 import tensor_to_base64
import random
import base64


#prompt生成
class TextGenerationNode():
    def __init__(self):
        super().__init__()
        # 从环境变量中获取 API 密钥
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system_prompt": ("STRING", {"default": "You are a helpful assistant."}),
                "user_prompt": ("STRING", {"default": "你是谁？"}),
                "top_p": ("FLOAT",{"default": 0.8,"min": 0.01, "max": 1.0}),
                "temperature" :("FLOAT", {"default": 1.0, "min": 0, "max": 1.9999}),
                "seed": ("INT",{"default": 0, "min": 0 , "max": 0xffffffffffffffff})
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_text"
    CATEGORY = "text_generation"

    def generate_text(self, system_prompt, user_prompt, top_p, temperature, seed):
        if not self.api_key:
            return ("错误: 未找到 API 密钥",)
        random.seed(seed)
        try:
            random.seed(seed)
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )

            completion = client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                top_p = top_p,
                temperature = temperature,
            )
            return (completion.choices[0].message.content,)
        except Exception as e:
            return (f"请求错误: {str(e)}",)

#图片理解
class ImageUnderstandingNode():
    def __init__(self):
        super().__init__()
        # 从环境变量中获取 API 密钥
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "system_prompt": ("STRING", {"default": "You are a helpful assistant."}),
                "user_prompt": ("STRING", {"default": "详细描述这张图片。"}),
                "top_p": ("FLOAT",{"default": 0.8,"min": 0.01, "max": 1.0}),
                "temperature" :("FLOAT", {"default": 1.0, "min": 0, "max": 1.9999}),
                "seed": ("INT",{"default": 0, "min": 0 , "max": 0xffffffffffffffff})
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "understand_image"
    CATEGORY = "image_understanding"

    def understand_image(self, image, system_prompt, user_prompt, top_p, temperature, seed):
        if not self.api_key:
            return ("错误: 未找到 API 密钥",)
        try:
            image_base64 = tensor_to_base64(image)
            random.seed(seed)
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )

            completion = client.chat.completions.create(
                model="qwen-vl-max",
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': [{
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                        "type": "text","text": user_prompt
                    }]}
                ],
                top_p = top_p,
                temperature = temperature,
            )
            return (completion.choices[0].message.content,)
        except Exception as e:
            return (f"请求错误: {str(e)}",)

NODE_CLASS_MAPPINGS = {
    "TextGeneration": TextGenerationNode,
    "ImageUnderstanding": ImageUnderstandingNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextGeneration": "Generate Text",
    "ImageUnderstanding": "Understand Image"
}