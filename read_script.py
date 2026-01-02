import os

target = r'\\192.168.2.126\Data\ComfyUI\custom_nodes\ray_killer.py'
output = r'\\192.168.2.126\Data\ComfyUI\custom_nodes\raylight\ray_killer_read.txt'

try:
    with open(target, 'r', encoding='utf-8') as f:
        content = f.read()
    with open(output, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Success")
except Exception as e:
    print(f"Error: {e}")
