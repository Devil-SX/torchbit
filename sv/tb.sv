`timescale 1ns/1ps
`include "reader.sv"


module tb  #(parameter BIT_WIDTH = `BIT_WIDTH) ;
  FileReader #(BIT_WIDTH) fr;
  FileCollecter #(BIT_WIDTH) fc;
  string input_file;
  string output_file;
  int continue_flag = 1;

  initial begin
    
    if ($value$plusargs("INPUT=%s", input_file) == 0) begin
      $display("Error: Input file not specified. Use +INPUT=filename.bin");
      $finish;
    end
    
    if ($value$plusargs("OUTPUT=%s", output_file) == 0) begin
      $display("Error: Output file not specified. Use +OUTPUT=filename.bin");
      $finish;
    end
    
    // FileReader #(.bit_width(BIT_WIDTH)) fr = new(input_file);
    // FileCollecter #(.bit_width(BIT_WIDTH)) fc = new(output_file);

    fr = new(input_file);
    fc = new(output_file);

    begin
      bit [BIT_WIDTH-1:0] data;
      while (continue_flag) begin
        continue_flag = fr.read_next(data);
        if (continue_flag) begin
          $display("Read data: 0x%x", data);
          fc.collect_data(data);
        end else begin
          $display("End of file reached or read error.");
        end
      end
      fc.dump();
      $finish;
    end
  end
endmodule
