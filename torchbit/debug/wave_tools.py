from wal.core import TraceContainer
from wal.eval import SEval
from wal.core import read_wal_sexpr

def analyze_valid_data(waveform_path, clk_name, valid_name, data_name):
    container = TraceContainer()
    container.load(waveform_path)
    
    evaluator = SEval(container)
    
    wal_query = f"""
    (map (lambda (t) (list t (at t {data_name}))) 
         (find (&& (= {clk_name} 1) (= (prev {clk_name}) 0) (= {valid_name} 1))))
    """
    
    try:
        parsed_expr = read_wal_sexpr(wal_query)
        result = evaluator.eval(parsed_expr)
        
        times = [item[0] for item in result]
        data_values = [item[1] for item in result]
        
        return times, data_values

    except Exception as e:
        print(f"Error during WAL analysis: {e}")
        return [], []
