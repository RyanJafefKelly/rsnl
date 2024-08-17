import matlab.engine
import multiprocessing as mp
import traceback

engines = {}

def start_matlab_engine():
    try:
        print('debug: start_matlab_engine')
        pid = mp.current_process().pid
        print('debug: mp.current_process().pid =', pid)
        eng = matlab.engine.start_matlab()
        eng.addpath('FUCCI')
        engines[pid] = eng
        print(f"MATLAB engine started for process {pid}")
    except Exception as e:
        print(f"Failed to start MATLAB engine in process {pid}: {str(e)}")
        traceback.print_exc()

def stop_matlab_engine():
    print('debug: stop_matlab_engine')
    print('debug: mp.current_process().pid =', mp.current_process().pid)
    eng = engines.pop(mp.current_process().pid, None)
    if eng:
        eng.quit()
