import math

if __name__ == "__main__":
    import os
    import time
    wait = int(os.environ.get("DEBUG_WAIT", "0"))
    if wait > 0:
        print(f"Waiting {wait}s for debugger to attach...", flush=True)
        time.sleep(wait)
    print(1 + 1, flush=True)