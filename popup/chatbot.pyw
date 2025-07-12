import customtkinter as ctk
import os
import time
from datetime import datetime

# --- Global constants / colors ---

BASE_DIR = os.path.dirname(__file__)
FILE_PATH = os.path.join(BASE_DIR, "text.txt")
USER_MSG_LOG = os.path.join(BASE_DIR, "user_messages.txt")  # Fixed: was tuple, now string
REFRESH_INTERVAL_MS = 500

BG_COLOR = "#1F2833"          # Dark blue-gray background
CHAT_BOT_BUBBLE = "#2C3E50"   # Slightly lighter dark for bot messages
CHAT_USER_BUBBLE = "#3B82F6"  # MS Teams blue for user messages
TEXT_COLOR = "#D1D5DB"        # Light gray text color
PLACEHOLDER_COLOR = "#6B7280" # Medium gray placeholder text

TEXT_SIZE = 13
FONT = "Segoe UI"

# --- Initialization for customtkinter ---
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# --- Helper functions ---

def load_paragraphs(path):
    if not os.path.exists(path):
        return ["(File not found)"]
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    return [p.strip() for p in content.split('\n\n') if p.strip()]

def type_text(label, full_text, index=0, delay=15):
    if index <= len(full_text):
        label.configure(text=full_text[:index])
        label.after(delay, type_text, label, full_text, index + 1, delay)

def display_paragraph(text, parent, align_right=False, typing=False):
    outer = ctk.CTkFrame(parent, fg_color="transparent", corner_radius=0)
    outer.pack(fill='x', pady=5, padx=5, anchor='e' if align_right else 'w')

    bubble = ctk.CTkFrame(
        master=outer,
        fg_color=CHAT_USER_BUBBLE if align_right else CHAT_BOT_BUBBLE,
        corner_radius=12
    )
    # Pack without fill='x' so bubble only takes needed space
    bubble.pack(anchor='e' if align_right else 'w', padx=10, pady=2)
    
    label = ctk.CTkLabel(
        master=bubble,
        text="" if typing else text,
        text_color=TEXT_COLOR,
        font=ctk.CTkFont(FONT, TEXT_SIZE),
        wraplength=400,  # This still limits max width
        justify="left",
        anchor="w"
    )
    label.pack(padx=10, pady=(8, 4), anchor="w")

    if typing and not align_right:
        # Start typing animation for bot message
        type_text(label, text)
    elif typing and align_right:
        # For user message, just show immediately
        label.configure(text=text)

    timestamp = datetime.now().strftime("%H:%M")
    time_label = ctk.CTkLabel(
        master=bubble,
        text=timestamp,
        font=ctk.CTkFont(FONT, 9, "bold"),
        text_color="#A9C4FF" if align_right else "gray",
        anchor='e'
    )
    time_label.pack(anchor='e', padx=10, pady=(0, 4))


def refresh_paragraphs(frame, canvas, state, seen):
    mtime = os.path.getmtime(FILE_PATH) if os.path.exists(FILE_PATH) else None
    if mtime != state[0]:
        paragraphs = load_paragraphs(FILE_PATH)
        for para in paragraphs[len(seen):]:
            display_paragraph(para, frame, align_right=False, typing=True)
            seen.append(para)
        canvas.configure(scrollregion=canvas.bbox("all"))
        state[0] = mtime
        canvas.after_idle(lambda: canvas.yview_moveto(1.0))
    canvas.after(REFRESH_INTERVAL_MS, refresh_paragraphs, frame, canvas, state, seen)

def log_user_message(message):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(USER_MSG_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{now}] {message}\n")

def add_placeholder(textbox, placeholder="Ask me anything..."):
    def on_focus_in(event):
        if textbox.get("1.0", "end-1c") == placeholder:
            textbox.delete("1.0", "end")
            textbox.configure(text_color=TEXT_COLOR)

    def on_focus_out(event):
        if textbox.get("1.0", "end-1c").strip() == "":
            textbox.insert("1.0", placeholder)
            textbox.configure(text_color=PLACEHOLDER_COLOR)

    textbox.insert("1.0", placeholder)
    textbox.configure(text_color=PLACEHOLDER_COLOR)
    textbox.bind("<FocusIn>", on_focus_in)
    textbox.bind("<FocusOut>", on_focus_out)
    return on_focus_out  # Return the function so we can call it manually

def on_mousewheel(event, canvas):
    canvas.yview_scroll(-1 * int(event.delta / 120), "units")

def on_canvas_resize(event, canvas, window_id):
    canvas.itemconfig(window_id, width=event.width)

def send_message(input_text, frame, canvas, root, restore_placeholder):
    msg = input_text.get("1.0", "end-1c").strip()
    if msg and msg != "Ask me anything...":
        display_paragraph(msg, frame, align_right=True)
        log_user_message(msg)
        input_text.delete("1.0", "end")
        # Force focus away from textbox, then restore placeholder
        root.focus_set()  # Move focus to root window
        restore_placeholder(None)
        root.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.yview_moveto(1.0)

# --- Main GUI setup ---

def main():
    root = ctk.CTk()
    root.title("Chatbot GUI")
    root.geometry("500x400")
    root.configure(fg_color=BG_COLOR)
    root.minsize(500, 300)
    root.maxsize(500, 300)

    container = ctk.CTkFrame(root, fg_color=BG_COLOR, corner_radius=0)
    container.place(relx=0, rely=0, relwidth=1, relheight=0.78)

    canvas = ctk.CTkCanvas(container, bg=BG_COLOR, highlightthickness=0)
    scrollbar = ctk.CTkScrollbar(container, command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)

    frame = ctk.CTkFrame(canvas, fg_color=BG_COLOR)
    frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    window_id = canvas.create_window((0, 0), window=frame, anchor="nw")

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    canvas.bind_all("<MouseWheel>", lambda e: on_mousewheel(e, canvas))
    canvas.bind("<Configure>", lambda e: on_canvas_resize(e, canvas, window_id))

    bottom = ctk.CTkFrame(root, fg_color=BG_COLOR)
    bottom.place(relx=0, rely=0.78, relwidth=1, relheight=0.22)

    input_text = ctk.CTkTextbox(
        bottom,
        font=ctk.CTkFont(FONT, TEXT_SIZE),
        wrap='word',
        height=50,
        text_color=TEXT_COLOR,
        fg_color=CHAT_BOT_BUBBLE,
        corner_radius=8
    )
    input_text.place(relx=0.02, rely=0.2, relwidth=0.82, relheight=0.6)

    restore_placeholder = add_placeholder(input_text)

    send_button = ctk.CTkButton(
        bottom,
        text="➤",
        font=ctk.CTkFont(FONT, 16),
        command=lambda: send_message(input_text, frame, canvas, root, restore_placeholder),
        width=45,
        height=45,
        corner_radius=9999,
        fg_color=CHAT_USER_BUBBLE,
        text_color="white"
    )
    send_button.place(relx=0.92, rely=0.5, anchor='center')

    input_text.bind("<Return>", lambda e: (send_message(input_text, frame, canvas, root, restore_placeholder), "break")[1] if not (e.state & 0x0001) else None)

    last_state = [None]
    seen = []
    refresh_paragraphs(frame, canvas, last_state, seen)


    # Position window in bottom-right corner
    root.update_idletasks()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = root.winfo_width()
    window_height = root.winfo_height()

    x = screen_width - window_width
    y = screen_height - window_height

    root.geometry(f"+{x}+{y}")
    root.mainloop()

if __name__ == "__main__":
    main()