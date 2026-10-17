from transformers import pipeline
import inspect

pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
src = inspect.getsource(pipe.__class__)
with open("E:/dw/pipe_src.txt", "w", encoding="utf-8") as f:
    f.write(src)
print("Done, class:", pipe.__class__.__name__)
