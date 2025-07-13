import subprocess
import os
import psutil
import sys

def launch_chatbot():
    script_path = os.path.join(os.path.dirname(__file__), "chatbot.pyw")
    return subprocess.Popen(["pythonw", script_path], shell=True)

def kill_chatbot(chatbot_instance):
    # Clean file containing text for chatbot:
    with open(os.path.join(os.path.dirname(__file__), "text.txt"), "w", encoding="utf-8"):
        pass

    # Straight up killing the chatbot instance doesn't get rid
    # of the GUI popup. Need to kill child 
    if chatbot_instance and chatbot_instance.poll() is None:
        pid = chatbot_instance.pid
        try:
            # Try to kill the process and all its children
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            
            # Kill children first
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            
            # Kill parent
            parent.terminate()
            
            # Wait for processes to terminate
            gone, alive = psutil.wait_procs(children + [parent], timeout=3)
            
            # Force kill any remaining processes
            for proc in alive:
                try:
                    proc.kill()
                except psutil.NoSuchProcess:
                    pass
                    
            print("Chatbot process and children terminated.")
            
        except psutil.NoSuchProcess:
            print("Chatbot process already terminated.")