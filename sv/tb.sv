`timescale 1ns/1ps
`include "reader.sv"

module tb  #(parameter BIT_WIDTH = 48) ;
  initial begin
    string input_file  = "dump.bin";
    string output_file = "random_binary_out.bin";
    
    // 初始化文件路径（这里直接使用硬编码路径）
    FileReader #(.bit_width(BIT_WIDTH)) fr = new(input_file);
    FileCollecter #(.bit_width(BIT_WIDTH)) fc = new(output_file);

    // 自动验证结果
    begin
      bit [BIT_WIDTH-1:0] data;
      fr.read_next(data);
      $display("Read data: 0x%x", data);
      $display("Read data: 0x%x", data[15:0]);
      $display("Read data: 0x%x", data[31:16]);

      // while (fr.read_next(data)) begin
      //   $display("Read data: 0x%x", data);
      // end
      $finish;
    end
  end
endmodule
