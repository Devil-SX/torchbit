utils for deep learning accelerator developing

# Features
- Convert a 2D tensor to a hex file readable by Verilog and vice versa
- Spatial computing port mapping specification
- Hardware specification boundary condition handling mechanism

# Usage

```
git clone https://github.com/Devil-SX/dlrtl.git
cd dlrtl
pip install -e .
```

# Example Flow

## with VCS / Seperated (Not Recommended)

Python
- gen activation and results
- (padding activation)
- save activation to memhex `dlrtl.Hensor.to_memhex`
- save results to pth `torch.save`

VCS
- load activation memhex `$readmemh`
- driver interface
- executation
- collect interface
- save results to memhex (User define)

Python
- load results from memhex `dlrtl.Hensor.from_memhex`
- (depadding results)
- load results from pth `torch.load`
- compare `dlrtl.compare`


## with Cocotb / Integrated (Recommended)

Python
- gen activation and results
- (loop)
- (padding activation)
- convert activation to int `dlrtl.Hensor.from_tensor` & `dlrtl.Hensor.to_cocotb`
- runing
- convert int to result `dlrtl.Hensor.from_cocotb`
- (depadding results)
- (merge)
- compare `dlrtl.compare`

## with Picker / Integrated / VSCode Debugpy Friendly (Recommended)

todo

