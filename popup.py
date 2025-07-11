import tkinter as tk
import threading
import sys
import time

def read_input(text_widget):
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        text_widget.insert(tk.END, line)
        text_widget.see(tk.END)  # Scroll to the end

def main():
    root = tk.Tk()
    root.title("Live Command Line Input")

    text_box = tk.Text(root, wrap='word', font=('Arial', 12), height=20, width=60)
    text_box.pack(padx=10, pady=10)

    # Start a background thread to read from stdin
    thread = threading.Thread(target=read_input, args=(text_box,), daemon=True)
    thread.start()

    root.mainloop()

if __name__ == "__main__":
    main()
