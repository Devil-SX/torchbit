# torchbit Documentation

Utilities for deep learning accelerator development with Cocotb verification.

## Overview

**torchbit** provides Python utilities for hardware verification and deep learning accelerator development:

- **Core**: Value conversion between PyTorch (Array/Matrix) and HDL (Logic/LogicSequence) formats
- **Tiling**: Tensor-to-LogicSequence mapping via TileMapping with einops rearrangement
- **Runner**: Simulator management (Verilator, VCS)
- **Tools**: Verification components (Buffer, Driver/Monitor)
- **Debug**: Waveform analysis tools

```{toctree}
---
maxdepth: 2
caption: User Guides
---

guide
```

```{toctree}
---
maxdepth: 2
caption: API Reference
---

api
```

## Quick Links

- [Installation](https://github.com/Devil-SX/torchbit#installation)
- [Examples](https://github.com/Devil-SX/torchbit/tree/main/examples)
- [GitHub Repository](https://github.com/Devil-SX/torchbit)

[中文文档](zh/ "Switch to Chinese")
