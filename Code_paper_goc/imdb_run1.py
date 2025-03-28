import json
import subprocess
import os
import time
import sys
import threading

def save_config(nDim, nb:bool, lr, randomSeed, earlyStoppingPatience, imdb_data_path, results_dir, suffix='', **kargs):
    nb_string = 'nb' if nb else 'full'
    config = {
        'filename': imdb_data_path,
        'nDim': nDim,
        'nb': nb,
        'n': 500,
        'minTf': 0,
        'lr': lr,
        'nEpoch': 40,
        'subSamp': nb,
        'nbA': 2,
        'nbB': 3,
        'randomSeed': randomSeed,
        'earlyStoppingPatience': earlyStoppingPatience,
        'vecPath': f'{results_dir}/imdb_vectors_{nb_string}_{lr:.0e}{suffix}.jsonl',
        'logPath': f'{results_dir}/imdb_log_{nb_string}_{lr:.0e}{suffix}.txt',
        'test': True,
        'Cs': [0.01, 0.1, 1, 10, 20, 100, 1000],
        'verbose': 3  # Increase verbosity level to maximum
    }
    config.update(kargs)
    with open('config.json', 'w') as f:
        json.dump(config, f)
    print(f"Configuration saved with {config['nEpoch']} epochs and patience {config['earlyStoppingPatience']}")
    return


# Function to continuously read from a pipe and print output
def stream_output(pipe, prefix=''):
    for line in iter(pipe.readline, ''):
        print(f"{prefix}{line.strip()}")
        sys.stdout.flush()


if __name__=="__main__":
    start_time = time.time()
    print("Starting IMDB training process...")
    
    imdb_data_path = "files_root/imdb_data.jsonl"
    results_dir = "imdb_trainsize_experiment"
    max_ind = 0
    
    print("Reading data to determine dimensions...")
    with open(imdb_data_path) as f:
        for line in f:
            item = json.loads(line)
            max_ind = max(max_ind, max(item["elementIds"]))
    nDim = max_ind + 1
    print(f"Dimension determined: {nDim}")
    
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
        print(f"Created results directory: {results_dir}")
    
    print("Compiling Java code...")
    compile_result = subprocess.run(
        ['javac', '-cp', 'dvscript;build/jars/gson-2.8.9.jar', '-d', 'build/classes', 'dvscript/dv/cosine/java/Run.java'],
        capture_output=True,
        text=True
    )
    print(f"Compilation {'successful' if compile_result.returncode == 0 else 'failed'}")
    if compile_result.stderr:
        print(f"Compilation errors: {compile_result.stderr}")

    random_seed = 22
    use_nb = False
    nepoch = 5  # Reduced even further to 5 for faster training
    early_stopping = 3  # Reduced even further to 3 for faster training
    
    print(f"Configuring with: epochs={nepoch}, early_stopping_patience={early_stopping}")
    save_config(nDim, use_nb, 1e-3, random_seed, early_stopping, imdb_data_path, results_dir, 
               suffix=f"_{nepoch}epoch_p{early_stopping}", nEpoch=nepoch)
    
    print("Starting Java training process...")
    print(f"Time elapsed before training: {time.time() - start_time:.2f} seconds")
    
    training_start = time.time()
    
    # Run Java process with real-time output streaming
    java_cmd = ['java', '-Xmx8g', '-cp', 'build/classes;build/jars/gson-2.8.9.jar', 'dv.cosine.java.Run', '-Dfile.encoding=UTF-8']
    print(f"Running command: {' '.join(java_cmd)}")
    
    # Use Popen to stream output in real-time
    process = subprocess.Popen(
        java_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1  # Line buffered
    )
    
    print("\n--- TRAINING OUTPUT ---")
    
    # Create threads to handle stdout and stderr separately
    stdout_thread = threading.Thread(target=stream_output, args=(process.stdout,))
    stderr_thread = threading.Thread(target=stream_output, args=(process.stderr, "ERROR: "))
    
    # Start threads
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    stdout_thread.start()
    stderr_thread.start()
    
    try:
        # Periodically print a status update if there's no output
        last_update = time.time()
        while process.poll() is None:
            # Add a timer to periodically show that the script is still running
            current_time = time.time()
            if current_time - last_update > 30:  # Show status every 30 seconds
                elapsed = current_time - training_start
                print(f"[Status] Process still running - {elapsed:.1f} seconds elapsed")
                last_update = current_time
            time.sleep(1)
    except KeyboardInterrupt:
        print("Process interrupted by user")
        process.terminate()
    
    # Wait for process to complete
    process.wait()
    
    print("--- END OF TRAINING OUTPUT ---\n")
    
    training_time = time.time() - training_start
    total_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Exit code: {process.returncode}")