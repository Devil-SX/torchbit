#!/bin/bash

# 清理旧文件
rm -rf obj_dir output.txt input.txt

# 生成测试输入文件
echo -e "1\n2\n3\n255" > input.txt

# 使用Verilator编译并运行
verilator --cc --exe -Wno-fatal --build tb.sv --main --trace

# 运行仿真
obj_dir/Vtb

# 清理中间文件
rm -rf obj_dir input.txt
