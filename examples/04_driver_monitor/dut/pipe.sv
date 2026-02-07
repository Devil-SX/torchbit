// Pipeline DUT for Driver/Monitor Example
//
// This module implements a simple parameterized pipeline with configurable
// width and delay. Used to demonstrate the Driver and Monitor classes.

module Pipe #(
    parameter WIDTH = 32,
    parameter DELAY = 4
)(
    input  logic             clk,
    input  logic             rst_n,
    input  logic             din_vld,
    input  logic [WIDTH-1:0] din,
    output logic             dout_vld,
    output logic [WIDTH-1:0] dout
);

    logic [WIDTH-1:0] data_pipe [DELAY-1:0];
    logic             vld_pipe  [DELAY-1:0];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int i = 0; i < DELAY; i++) begin
                vld_pipe[i] <= 1'b0;
                data_pipe[i] <= '0;
            end
        end else begin
            vld_pipe[0] <= din_vld;
            data_pipe[0] <= din;
            for (int i = 1; i < DELAY; i++) begin
                vld_pipe[i] <= vld_pipe[i-1];
                data_pipe[i] <= data_pipe[i-1];
            end
        end
    end

    assign dout_vld = vld_pipe[DELAY-1];
    assign dout = data_pipe[DELAY-1];

endmodule
