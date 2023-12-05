import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

top_package_path = os.path.abspath(os.path.join(current_dir, "..","..",".."))

sys.path.insert(0, top_package_path)

model_root = os.path.join(top_package_path, "models")
print("model root:",model_root)


message = {
    "from":"keyboard",
    "to":None,
    "format":"text",
    "data": ""
}