// Tile Mover DUT - Transposes data from (b, w, c) layout to (w, b, c) layout
//
// This module performs a tile transpose operation:
// - Source layout: (b, w, c) * (cs) - (batch, width, channel) order
// - Destination layout: (w, b, c) * (cs) - transposed layout (b and w swapped)
// - cs: spatial dimension (elements transferred per cycle, e.g., cs=4 for 64bit/16bit)
//
// The dimensions (b, w, c) and base addresses are configurable via instruction port.
//
// TwoPortBuffer interface (from torchbit):
// - wr_csb: Write chip select (active low)
// - wr_din: Write data input
// - wr_addr: Write address
// - rd_csb: Read chip select (active low)
// - rd_addr: Read address
// - rd_dout: Read data output
// - rd_dout_vld: Read data valid

module tile_mover #(
    parameter ADDR_WIDTH = 8,
    parameter DATA_WIDTH = 64
)(
    input  wire clk,
    input  wire rst_n,

    // Instruction port - runtime configurable
    input  wire [ADDR_WIDTH-1:0] src_base_addr,
    input  wire [ADDR_WIDTH-1:0] dst_base_addr,
    input  wire [7:0] b_dim,         // B dimension (batch in source)
    input  wire [7:0] w_dim,         // W dimension (width in source)
    input  wire [7:0] c_dim,         // C dimension (channel, = ct * cs)
    input  wire start,
    output reg  done,

    // TwoPortBuffer connection
    output wire [ADDR_WIDTH-1:0] rd_addr,
    output wire rd_csb,
    input  wire [DATA_WIDTH-1:0] rd_dout,
    input  wire rd_dout_vld,

    output wire [ADDR_WIDTH-1:0] wr_addr,
    output wire [DATA_WIDTH-1:0] wr_din,
    output wire wr_csb
);

    // State machine
    typedef enum logic [2:0] {
        IDLE,
        READ_ELEM,
        WRITE_ELEM,
        COMPLETE
    } state_t;

    state_t state;
    logic [15:0] elem_count;     // Total element counter (b * w * ct)
    logic [7:0] b_idx;           // Current batch index
    logic [7:0] w_idx;           // Current width index
    logic [7:0] ct_idx;          // Current temporal channel index
    logic [7:0] ct_dim;          // Temporal channel dimension (c_dim / cs)
    logic [ADDR_WIDTH-1:0] src_addr;
    logic [ADDR_WIDTH-1:0] dst_addr;
    logic [DATA_WIDTH-1:0] read_data;

    // Spatial dimension (cs) - fixed for this design
    // For DATA_WIDTH=64 and 16-bit elements, cs=4
    localparam integer CS = 4;

    // Calculate ct = c / cs
    assign ct_dim = c_dim / CS;

    // Total elements = b * w * ct (each element contains cs spatial data)
    wire [15:0] total_elements;
    assign total_elements = b_dim * w_dim * ct_dim;

    // Source address calculation:
    // Layout: (b, w, ct) with cs spatial elements per address
    // addr = src_base + (b * w * ct + w * ct + ct) * 1 (linear index)
    assign src_addr = src_base_addr + (b_idx * w_dim * ct_dim) + (w_idx * ct_dim) + ct_idx;

    // Destination address calculation:
    // Layout: (w, b, ct) with cs spatial elements per address (b and w swapped!)
    // addr = dst_base + (w * b * ct + b * ct + ct) * 1 (linear index)
    assign dst_addr = dst_base_addr + (w_idx * b_dim * ct_dim) + (b_idx * ct_dim) + ct_idx;

    // Control signals - connect to TwoPortBuffer
    assign rd_addr = src_addr;
    assign rd_csb = (state != READ_ELEM);  // Active low: 0 = enabled

    assign wr_addr = dst_addr;
    assign wr_din = read_data;
    assign wr_csb = (state != WRITE_ELEM);  // Active low: 0 = enabled

    // FSM and counters
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            elem_count <= 0;
            b_idx <= 0;
            w_idx <= 0;
            ct_idx <= 0;
            done <= 0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 0;
                    if (start) begin
                        state <= READ_ELEM;
                        elem_count <= 0;
                        b_idx <= 0;
                        w_idx <= 0;
                        ct_idx <= 0;
                    end
                end

                READ_ELEM: begin
                    // Read cycle - wait for valid data
                    if (rd_dout_vld) begin
                        // Buffer the read data
                        read_data <= rd_dout;
                        state <= WRITE_ELEM;
                    end
                end

                WRITE_ELEM: begin
                    // Write cycle - data is registered, move to next element
                    if (elem_count < total_elements - 1) begin
                        elem_count <= elem_count + 1;

                        // Update indices: b, w, ct nested loops
                        if (ct_idx < ct_dim - 1) begin
                            ct_idx <= ct_idx + 1;
                        end else begin
                            ct_idx <= 0;
                            if (w_idx < w_dim - 1) begin
                                w_idx <= w_idx + 1;
                            end else begin
                                w_idx <= 0;
                                if (b_idx < b_dim - 1) begin
                                    b_idx <= b_idx + 1;
                                end else begin
                                    b_idx <= 0;
                                end
                            end
                        end

                        state <= READ_ELEM;
                    end else begin
                        // All elements processed
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

endmodule
