import tkinter as tk
import os
import time
from datetime import datetime

FILE_PATH = "text.txt"
LOG_FILE = "record_log.txt"
USER_MSG_LOG = "user_messages.txt"
REFRESH_INTERVAL_MS = 500

def load_paragraphs(path):
    if not os.path.exists(path):
        return ["(File not found)"]
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    return [p.strip() for p in content.split('\n\n') if p.strip()]

def display_paragraph(text, parent, align_right=False):
    outer = tk.Frame(parent, bg='lightgray')
    outer.pack(fill='x', pady=5, padx=5)

    bubble = tk.Frame(
        outer,
        bg="#dcf8c6" if align_right else "white",
        highlightbackground="black",
        highlightthickness=1
    )

    label = tk.Label(
        bubble,
        text=text,
        wraplength=400,
        justify='left',
        anchor='w',
        font=('Arial', 10),
        bg=bubble['bg']
    )
    label.pack(padx=10, pady=8)

    bubble.pack(side='right' if align_right else 'left', padx=10)

def refresh_paragraphs(frame, canvas, state, seen):
    mtime = os.path.getmtime(FILE_PATH) if os.path.exists(FILE_PATH) else None
    if mtime != state[0]:
        paragraphs = load_paragraphs(FILE_PATH)
        for para in paragraphs[len(seen):]:
            display_paragraph(para, frame, align_right=False)
            seen.append(para)
        canvas.configure(scrollregion=canvas.bbox("all"))
        state[0] = mtime
        canvas.after_idle(lambda: canvas.yview_moveto(1.0))  # Scroll to bottom
    canvas.after(REFRESH_INTERVAL_MS, refresh_paragraphs, frame, canvas, state, seen)

def log_recording_event(event_type):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{event_type} recording at {now}\n")

def log_user_message(message):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(USER_MSG_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{now}] {message}\n")

def main():
    root = tk.Tk()
    root.title("Text Viewer")
    root.geometry("500x400")
    root.minsize(500, 300)

    container = tk.Frame(root)
    container.place(relx=0, rely=0, relwidth=1, relheight=0.72)

    canvas = tk.Canvas(container, bg='lightgray')
    scrollbar = tk.Scrollbar(container, command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)

    frame = tk.Frame(canvas, bg='lightgray')
    frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    window_id = canvas.create_window((0, 0), window=frame, anchor="nw")

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    def on_canvas_resize(event):
        canvas.itemconfig(window_id, width=event.width)

    canvas.bind("<Configure>", on_canvas_resize)

    bottom = tk.Frame(root, bg='gray')
    bottom.place(relx=0, rely=0.72, relwidth=1, relheight=0.28)

    input_text = tk.Text(bottom, font=('Arial', 10), wrap='word')
    input_text.place(relx=0.01, rely=0.05, relwidth=0.65, relheight=0.9)

    is_recording = [False]
    start_time = [None]

    def update_record_button_text():
        if is_recording[0]:
            elapsed = int(time.time() - start_time[0])
            mins = elapsed // 60
            secs = elapsed % 60
            record_button.config(text=f"Stop Recording ({mins:02d}:{secs:02d})")
            root.after(500, update_record_button_text)
        else:
            record_button.config(text="Start Recording")

    def toggle_recording():
        if not is_recording[0]:
            is_recording[0] = True
            start_time[0] = time.time()
            log_recording_event("Start")
            update_record_button_text()
        else:
            is_recording[0] = False
            log_recording_event("Stop")
            record_button.config(text="Start Recording")

    def send_message():
        msg = input_text.get("1.0", "end-1c").strip()
        if msg:
            display_paragraph(msg, frame, align_right=True)
            log_user_message(msg)
            input_text.delete("1.0", "end")

            root.update_idletasks()
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.yview_moveto(1.0)

    input_text.bind("<Return>", lambda e: (send_message(), "break")[1] if not (e.state & 0x0001) else None)

    send_button = tk.Button(bottom, text="Send", font=('Arial', 10), command=send_message)
    send_button.place(relx=0.68, rely=0.05, relwidth=0.3, relheight=0.45)

    record_button = tk.Button(bottom, text="Start Recording", font=('Arial', 10), command=toggle_recording)
    record_button.place(relx=0.68, rely=0.52, relwidth=0.3, relheight=0.45)

    last_state = [None]
    seen = []
    refresh_paragraphs(frame, canvas, last_state, seen)

    root.mainloop()

if __name__ == "__main__":
    main()
