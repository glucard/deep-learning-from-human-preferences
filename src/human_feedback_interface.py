import tkinter as tk
from tkinter import filedialog
import cv2
import threading
from PIL import Image, ImageTk
import time, queue

class VideoPicker:
    def __init__(self, master):
        self.master = master
        self.master.title("HumanFeedback Interface")
        # Set fixed position and size
        self.master.geometry('1200x300+100+100')  # width x height + x_offset + y_offset

        self.feedback = tk.StringVar()

        self.video1_path = None
        self.video2_path = None

        self.canvas1 = tk.Canvas(master, width=400, height=300)
        self.canvas1.pack(side=tk.LEFT)

        self.canvas2 = tk.Canvas(master, width=400, height=300)
        self.canvas2.pack(side=tk.RIGHT)

        self.btn_select_video1 = tk.Button(master, text="left", command=lambda: self.pick_video('1'))
        self.btn_select_video1.pack(side=tk.LEFT)

        self.btn_select_video2 = tk.Button(master, text="right", command=lambda: self.pick_video('2'))
        self.btn_select_video2.pack(side=tk.RIGHT)

        self.btn_select_video2 = tk.Button(master, text="equals", command=lambda: self.pick_video('e'))
        self.btn_select_video2.pack(side=tk.TOP)
        
        self.btn_select_video2 = tk.Button(master, text="incomparable", command=lambda: self.pick_video('n'))
        self.btn_select_video2.pack(side=tk.BOTTOM)

        self.thread1 = None
        self.thread2 = None

        self.stop_event = threading.Event()
        
        self.lock = threading.Lock()
        self.queue1 = queue.Queue()
        self.queue2 = queue.Queue()
        

        self.stop_event = threading.Event()  # Create an event to signal threads to stop

        self.update_frame()

        # Store PhotoImage references to prevent garbage collection
        self.img_tk1 = None
        self.img_tk2 = None
        
        self.load_video1()  
        self.load_video2()

    def load_video1(self):
        self.video1_path = "1.avi"
        if self.video1_path:
            if self.thread1 and self.thread1.is_alive():
                self.stop_event.set()
                self.thread1.join()
                self.stop_event.clear()
            self.thread1 = threading.Thread(target=self.play_video, args=(self.video1_path, self.queue1))
            self.thread1.start()

    def load_video2(self):
        self.video2_path = "2.avi"
        if self.video2_path:
            if self.thread2 and self.thread2.is_alive():
                self.stop_event.set()
                self.thread2.join()
                self.stop_event.clear()
            self.thread2 = threading.Thread(target=self.play_video, args=(self.video2_path, self.queue2))
            self.thread2.start()

    def play_video(self, video_path, frame_queue):
        print(f"Playing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = 1 / fps  # Delay to match the video FPS
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (400, 300))
            frame_queue.put(frame)
            time.sleep(delay)  # Add delay to control frame rate
        cap.release()
        print(f"Stopped playing video: {video_path}")

    def update_frame(self):
        if not self.queue1.empty():
            frame = self.queue1.get()
            img = Image.fromarray(frame)
            if self.img_tk1 is None:
                self.img_tk1 = ImageTk.PhotoImage(image=img)
                self.canvas1.create_image(0, 0, anchor=tk.NW, image=self.img_tk1)
            else:
                self.img_tk1.paste(img)
                self.canvas1.create_image(0, 0, anchor=tk.NW, image=self.img_tk1)

        if not self.queue2.empty():
            frame = self.queue2.get()
            img = Image.fromarray(frame)
            if self.img_tk2 is None:
                self.img_tk2 = ImageTk.PhotoImage(image=img)
                self.canvas2.create_image(0, 0, anchor=tk.NW, image=self.img_tk2)
            else:
                self.img_tk2.paste(img)
                self.canvas2.create_image(0, 0, anchor=tk.NW, image=self.img_tk2)

        self.master.after(10, self.update_frame)  # Schedule the next frame update
        
    def pick_video(self, choice:str) -> str:
        self.feedback.set(choice)
        self.on_closing()
        self.master.quit()  # Stop the main loop
        # self.master.destroy()
        

    def on_closing(self):

        self.stop_event.set()
        if self.thread2 and self.thread2.is_alive():
            print("closing2")
            self.thread2.join()
            print("closed2")
        if self.thread1 and self.thread1.is_alive():
            print("closing1")
            self.thread1.join()
            print("closed1")
        # self.stop_event.clear()
        # self.master.destroy()

        

def interface_pick():
    
    root = tk.Tk()
    app = VideoPicker(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
    feedback_value = app.feedback.get()
    root.destroy()

    time.sleep(0.1)
    return feedback_value

if __name__ == "__main__":
    print(interface_pick())