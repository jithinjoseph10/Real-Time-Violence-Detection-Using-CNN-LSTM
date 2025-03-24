# gui.py
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from detection import real_time_detection, upload_video

def resize_background(event):
    canvas.delete("gradient")
    canvas.create_rectangle(0, 0, event.width, event.height, fill=gradient_color1, width=0, tags="gradient")
    for i in range(event.height):
        canvas.create_line(0, i, event.width, i, fill=gradient_color2, width=1, tags="gradient")

# Create the main window
window = tk.Tk()
window.title("VIOLENCE DETECTION")

# Create a canvas widget for the background gradient
canvas = tk.Canvas(window)
canvas.pack(fill="both", expand=True)

# Define the gradient colors
gradient_color1 = "#8ED6FF"  # Light blue
gradient_color2 = "#004CB5"  # Dark blue

# Apply a custom style with light colors for buttons and labels
style = ttk.Style()
style.configure("Custom.TButton", background="#D6E6F5", foreground="#333333", font=("Helvetica", 12, "bold"), relief="flat", borderwidth=0)
style.configure("Custom.TLabel", background="#D6E6F5", foreground="#333333", font=("Helvetica", 14))

# Create a frame to hold the buttons and labels
frame = tk.Frame(window, bg="#D6E6F5")
frame.place(relx=0.5, rely=0.5, anchor="center")

# Create a label widget with a custom style
label = ttk.Label(frame, text="Choose an option:", style="Custom.TLabel")
label.pack(pady=10)

# Create a frame to hold the buttons
button_frame = tk.Frame(frame, bg="#D6E6F5")
button_frame.pack()

# Create a button for real-time detection with a custom style
realtime_button = ttk.Button(button_frame, text="Real-Time Detection", command=real_time_detection, style="Custom.TButton")
realtime_button.pack(side="left", padx=5)

# Create a button for video upload with a custom style
upload_button = ttk.Button(button_frame, text="Upload Video", command=upload_video, style="Custom.TButton")
upload_button.pack(side="left", padx=5)

# Bind the resizing event of the canvas to adjust the background gradient
canvas.bind("<Configure>", resize_background)

# Start the main event loop
window.mainloop()
