# torchbit 文档

深度学习加速器开发与 Cocotb 验证工具库。

## 概述

**torchbit** 提供了硬件验证和深度学习加速器开发的 Python 工具：

- **Core** (核心): PyTorch 与 HDL 格式之间的张量/向量转换
- **Tiling** (分片): 张量到内存映射，支持空���/时间维度
- **Runner** (运行器): 仿真器管理 (Verilator, VCS)
- **Tools** (工具): 验证组件 (Buffer, Driver/Monitor)
- **Debug** (调试): 波形分析工具

```{toctree}
---
maxdepth: 2
caption: 用户指南
---

../guide
```

```{toctree}
---
maxdepth: 2
caption: API 参考
---

../api
```

## 快���链接

- [安装说明](https://github.com/Devil-SX/torchbit#installation)
- [示例](https://github.com/Devil-SX/torchbit/tree/main/examples)
- [GitHub 仓库](https://github.com/Devil-SX/torchbit)

[English](../ "Switch to English")
