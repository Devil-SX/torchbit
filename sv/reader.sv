
// https://www.chipverify.com/systemverilog/systemverilog-parameterized-classes
class FileReader #(
    int bit_width = 8
);
  local int pointer = 0;
  local logic [bit_width-1:0] data[$];

  function new(string file_path);
    int fd;
    fd = $fopen(file_path, "r");
    if (fd) begin
      int temp;
      while (!$feof(
          fd
      )) begin
        logic [bit_width-1:0] temp;
        if ($fread(temp, fd) == bit_width / 8) begin
          this.data.push_back(temp); 
        end
      end
      $fclose(fd);
    end
  endfunction

  function int read_next(output logic [bit_width-1:0] out_data);
    if (this.pointer < this.data.size()) begin
      out_data = this.data[this.pointer];
      this.pointer++;
      return 1;  // Success
    end else begin
      out_data = '0;
      return 0;  // Failure
    end
  endfunction
endclass


class FileCollecter #(
    int bit_width = 8
);

  local string file_path;
  local logic [bit_width-1:0] data_queue[$];

  function new(string file_path);
    this.file_path = file_path;
  endfunction

  function void collect_data(logic [bit_width-1:0] in_data);
    data_queue.push_back(in_data);
  endfunction

  function int dump();
    int fd = $fopen(file_path, "w");
    if (fd == 0) begin
      $display("[Error] Failed to open file: %s", file_path);
      return 0; // Failure
    end
    foreach (data_queue[i]) begin
      $fwrite(fd, "%c", data_queue[i]);
    end
    $fclose(fd);
    $display("[Info] Dumped %0d lines to %s", data_queue.size(), file_path);
    return 1; // Success
  endfunction
endclass
