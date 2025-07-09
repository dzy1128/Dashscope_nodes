import base64
import torch
from PIL import Image
import io

def tensor_to_base64(tensor_image):
    """
    将 ComfyUI 中的 PyTorch 张量格式图像转换为 Base64 编码字符串
    尽可能长时间地保持数据在 GPU 上
    
    Args:
        tensor_image: PyTorch 张量格式的图像，形状为 [1, H, W, 3] 或 [N, H, W, 3]，值范围 0-1
        
    Returns:
        base64_string: Base64 编码的字符串
    """
    # 如果有批次维度，取第一张图片 (在 GPU 上操作)
    if len(tensor_image.shape) == 4:
        if tensor_image.shape[0] > 1:
            # 多张图片时取第一张
            image_tensor = tensor_image[0]
        else:
            # 单张图片
            image_tensor = tensor_image.squeeze(0)
    else:
        image_tensor = tensor_image
    
    # 在 GPU 上将值从 0-1 范围转换为 0-255
    image_tensor = (image_tensor * 255).round()
    
    # 只有在真正需要转换格式时才将数据移至 CPU
    numpy_image = image_tensor.cpu().numpy().astype('uint8')
    
    # 创建 PIL 图像
    pil_image = Image.fromarray(numpy_image)
    
    # 创建一个字节流缓冲区
    buffer = io.BytesIO()
    
    # 将图像保存到缓冲区，格式为 PNG
    pil_image.save(buffer, format="PNG")
    
    # 获取字节数据并进行 Base64 编码
    img_bytes = buffer.getvalue()
    base64_data = base64.b64encode(img_bytes)
    
    # 转换为字符串并返回
    base64_string = base64_data.decode('utf-8')
    return base64_string
