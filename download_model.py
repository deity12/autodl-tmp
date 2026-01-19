# download_model.py
from modelscope import snapshot_download

print("开始下载 Qwen2.5-14B-Instruct 模型...")
print("文件较大（约 28GB），请耐心等待，通常速度 50MB/s 以上...")

# 这里的 cache_dir 非常重要！
# 一定要指向 /root/autodl-tmp/，否则你的系统盘瞬间就爆了
model_dir = snapshot_download(
    'qwen/Qwen2.5-14B-Instruct',
    cache_dir='/root/autodl-tmp/models',
    revision='master'
)

print(f"下载成功！模型保存路径为: {model_dir}")
print("你可以把这个路径复制下来，后面写代码要用到。")