import os, sys
B = r"e:/dw/terravue"
def w(rel, txt):
    p = os.path.join(B, rel.replace("/", os.sep))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p,"w",encoding="utf-8").write(txt)
    print("wrote", rel)
