"""RADAR 上游代码的 vendor 副本（上游许可头随文件保留在各自源码里）。

为什么需要这个 __init__：`vision_branch.py:29` 用
`pydoc.locate("dynamic_network_architectures.architectures.unet_lightdecoder.PlainConvUNetLightD")`
**按字符串反射**取网络类，所以 `dynamic_network_architectures` 必须能从 sys.path 顶层 import 到。
这里只把 vendor/ 目录挂进 sys.path，不做别的 import（静态 import 探测查不出这个问题，
见 plan 步骤 0 的实测）。
"""
import os
import sys

_VENDOR_DIR = os.path.dirname(os.path.abspath(__file__))
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)
