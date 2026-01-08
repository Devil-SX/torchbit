// Memory Mover DUT - Copies data from source to destination region
//
// This module copies data from a source memory region to a destination region
// using TWO TwoPortBuffer instances:
// - src_buffer: For reading source data (uses read port)
// - dst_buffer: For writing destination data (uses write port)
//
// TwoPortBuffer interface (from torchbit):
// - wr_csb: Write chip select (active low)
// - wr_din: Write data input
// - wr_addr: Write address
// - rd_csb: Read chip select (active low)
// - rd_addr: Read address
// - rd_dout: Read data output
// - rd_dout_vld: Read data valid

module memory_mover #(
    parameter ADDR_WIDTH = 8,
    parameter DATA_WIDTH = 32
)(
    input  wire clk,
    input  wire rst_n,

    // Control interface
    input  wire start,                  // Start the copy operation
    input  wire [ADDR_WIDTH-1:0] src_base_addr,   // Source region base address
    input  wire [ADDR_WIDTH-1:0] dst_base_addr,   // Destination region base address
    input  wire [7:0] num_words,         // Number of words to copy
    output reg  done,                    // Copy complete flag

    // Source Buffer (TwoPortBuffer - read port only)
    output wire [ADDR_WIDTH-1:0] src_rd_addr,
    output wire src_rd_csb,
    input  wire [DATA_WIDTH-1:0] src_rd_dout,
    input  wire src_rd_dout_vld,

    // Destination Buffer (TwoPortBuffer - write port only)
    output wire [ADDR_WIDTH-1:0] dst_wr_addr,
    output wire [DATA_WIDTH-1:0] dst_wr_din,
    output wire dst_wr_csb
);

    // State machine
    typedef enum logic [1:0] {
        IDLE,
        READ_SRC,
        WRITE_DST,
        COMPLETE
    } state_t;

    state_t state;
    logic [7:0] count;
    logic [ADDR_WIDTH-1:0] src_addr;
    logic [ADDR_WIDTH-1:0] dst_addr;
    logic [DATA_WIDTH-1:0] read_data;

    // Address and control signals - connect to TwoPortBuffer
    assign src_rd_addr = src_addr;
    assign src_rd_csb = (state != READ_SRC);  // Active low: 0 = enabled

    assign dst_wr_addr = dst_addr;
    assign dst_wr_din = read_data;
    assign dst_wr_csb = (state != WRITE_DST);  // Active low: 0 = enabled

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            count <= 0;
            src_addr <= 0;
            dst_addr <= 0;
            done <= 0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 0;
                    if (start) begin
                        state <= READ_SRC;
                        count <= 0;
                        src_addr <= src_base_addr;
                        dst_addr <= dst_base_addr;
                    end
                end

                READ_SRC: begin
                    // Read cycle - wait for valid data
                    if (src_rd_dout_vld) begin
                        state <= WRITE_DST;
                    end
                end

                WRITE_DST: begin
                    // Write cycle - data is already registered
                    if (count < num_words - 1) begin
                        count <= count + 1;
                        src_addr <= src_addr + 1;
                        dst_addr <= dst_addr + 1;
                        state <= READ_SRC;
                    end else begin
                        state <= COMPLETE;
                        done <= 1;
                    end
                end

                COMPLETE: begin
                    if (!start) begin
                        state <= IDLE;
                        done <= 0;
                    end
                end

                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end

    // Capture read data (registered for timing)
    always_ff @(posedge clk) begin
        if (src_rd_dout_vld) begin
            read_data <= src_rd_dout;
        end
    end

endmodule
