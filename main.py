import gc
import time
import sys
import tkinter as tk
import threading
from Detector import Detector
from PIL import Image, ImageTk
from pynput import keyboard
from robomaster import robot

from ultralytics import YOLO
from torch.cuda import is_available as cuda_available

device = 'cuda' if cuda_available() else 'cpu'
ep_camera = None
ep_robot = None
detector = None
buttons = []
panel = None
yaw=90
pitch=15

stop_event = threading.Event()
def rotate_camera(angle):
    global ep_robot
    global yaw
    global pitch
    yaw=yaw+angle
    if yaw>180:
        yaw=180
    if yaw<-180:
        yaw=-180
    ep_robot.gimbal.moveto(pitch,yaw).wait_for_completed()

def on_press(key):
    global ep_robot
    
    if ep_robot!=None and key!=None:
        if key.char == 'w':
            ep_robot.chassis.move(x=0.5, y=0, z=0, xy_speed=0.2).wait_for_completed()
        elif key.char == 's':
            ep_robot.chassis.move(x=-0.5, y=0, z=0, xy_speed=0.2).wait_for_completed()
        elif key.char == 'a':
            ep_robot.chassis.move(x=0, y=-0.5, z=0, xy_speed=0.2).wait_for_completed()
        elif key.char == 'd':
            ep_robot.chassis.move(x=0, y=0.5, z=0, xy_speed=0.2).wait_for_completed()
        elif key.char == 'q':
            rotate_camera(-10)  
        elif key.char == 'e':
            rotate_camera(10)


def keyboard_listener():
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


def stop_connection(buttons):
    global ep_camera
    global detector
    global panel
    if detector is not None:
        detector.stop_stream = True
    if ep_camera is not None:
        ep_camera.stop_video_stream()
    for button in buttons:
        button.config(state=tk.NORMAL)
    detector=None
    image = Image.open('background.png')
    imgtk = ImageTk.PhotoImage(image)
    panel.imgtk = imgtk
    panel.config(image=imgtk)
    gc.collect()


def create_detector(ep_robot, nn, panel, buttons, device='cpu'):
    for button in buttons:
        button.config(state=tk.DISABLED)
    global detector
    detector = Detector(ep_robot, nn, panel, device)
    stream = threading.Thread(target=detector.start_stream)
    stream.start()
    #del detector
    gc.collect()


def change_window_content():
    global ep_camera
    global detector
    global device
    global ep_robot
    global panel
    for widget in main_window.winfo_children():
        widget.destroy()
    try:
        ep_robot = robot.Robot()
        ep_robot.initialize()
        ep_camera = ep_robot.camera
        print('Соединение установлено')
    except Exception:
        print('Соединение не установлено')
        #sys.exit()
    
    time.sleep(1)
    button1 = tk.Button(main_window, text="YOLO8n ", font=("Arial", 16), width=15,
                        command=lambda: create_detector(ep_robot, 'yolo', panel, buttons, device))
    button2 = tk.Button(main_window, text="yolo custom", font=("Arial", 16), width=15,
                        command=lambda: create_detector(ep_robot, 'custom', panel, buttons, device))
    button3 = tk.Button(main_window, text="mobilenet ssd", font=("Arial", 16), width=15,
                        command=lambda: create_detector(ep_robot, 'mobilenet', panel, buttons, device))
    panel = tk.Label(main_window)
    panel.pack(pady=10)

    for button in [button1, button2, button3]:
        button.pack(side=tk.LEFT, anchor=tk.SE)
        buttons.append(button)
        if ep_camera is None:
            button.config(state=tk.DISABLED)

    image = Image.open('background.png')
    imgtk = ImageTk.PhotoImage(image)
    panel.imgtk = imgtk
    panel.config(image=imgtk)

    disconnect_button = tk.Button(main_window, text="X", font=("Arial", 16), width=20,
                                  command=lambda: stop_connection(buttons))
    disconnect_button.pack(side=tk.BOTTOM, anchor=tk.SE)


main_window = tk.Tk()
main_window.title("Главное окно")
main_window.geometry("800x600")

connect_button = tk.Button(main_window, text="Подключиться", font=("Arial", 16), width=20,
                           command=change_window_content)
connect_button.pack(pady=40)
keyboard_thread = threading.Thread(target=keyboard_listener)
keyboard_thread.start()
while not stop_event.is_set():
    main_window.mainloop()
#keyboard_thread.join()
