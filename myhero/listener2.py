'''
from pynput import mouse
import time

# --- functions ---

def do_work(x, y, button, pressed):
    if pressed:
        # emulate long-running process
        print("Started work.")
        time.sleep(3)
        print("Finished work.")

# --- main ---

with mouse.Listener(on_click=do_work) as listener:
    # ... other code ...
    listener.join()
'''

from pynput import mouse
import threading
import time

# --- functions ---

def do_work(x, y, button, pressed):
    # emulate long-running process
    print("Started work.")
    time.sleep(3)
    print("Finished work.")

def on_click(x, y, button, pressed):
    global job

    if pressed:
        if job is None or not job.is_alive():
            job = threading.Thread(target=do_work, args=(x, y, button, pressed))
            job.start()
        #else:
        #    print("skiping this click")

# --- main ---

job = None  # default value at start

with mouse.Listener(on_click=on_click) as listener:
    # ... other code ...
    listener.join()
