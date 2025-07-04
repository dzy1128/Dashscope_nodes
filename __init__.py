from .dashscope_nodes import TextGenerationNode


NODE_CLASS_MAPPINGS = {
    "TextGeneration": TextGenerationNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextGeneration": "Generate Text"
}