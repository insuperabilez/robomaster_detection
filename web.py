import streamlit as st
import time
import robomaster
import time
import cv2
from ultralytics import YOLO
from robomaster import conn, robot
from MyQR import myqr
from PIL import Image

QRCODE_NAME = "qrcode.png"

def app():
    login = st.text_input('Введите LOGIN')
    password = st.text_input('Введите PASSWORD', type="password")
    serial_number = st.text_input('Введите серийный номер')
    model=YOLO("yolo-Weights/custom.pt")
    if st.button('Подключиться'):
        helper = conn.ConnectionHelper()
        info = helper.build_qrcode_string(ssid=login, password=password)
        myqr.run(words=info)
        time.sleep(1)
        img = Image.open(QRCODE_NAME)
        #img.show()
        st.image(img)
        if helper.wait_for_connection():
            st.write("Подключено!")
            time.sleep(10)

            ep_robot = robot.Robot()
            ep_robot.initialize(conn_type='sta',sn=serial_number)

            # подключаемся к камере
            ep_camera = ep_robot.camera
            ep_camera.start_video_stream(display=False)

            # забираем изображение с камеры
            disconnect_button = st.button('Отключение')
            
            while not disconnect_button:
                # забираем изображение с камеры
                img = ep_camera.read_cv2_image(strategy='newest')
                results=model(img)
                for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    print("Confidence --->", confidence)

                    cls = int(box.cls[0])
                    print("Class name -->", model.names[cls])

                    org = [x1, y1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 0.5
                    color = (0, 255, 0)
                    thickness = 2
                    cv2.putText(img, model.names[cls], org, font, fontScale, color, thickness)
                    cv2.putText(img, str(confidence), [x2, y1 + 10], font, fontScale, color, thickness)
                # преобразуем изображение в формат, подходящий для Streamlit
                frame = Image.fromarray(img)

                # выводим изображение на страницу Streamlit
                #st.image(frame)
                imageLocation.image(frame)
            # закрываем соединение с камерой и роботом
            ep_camera.stop_video_stream()
            ep_robot.close()

        else:
            st.write("Подключение не удалось!")

if __name__ == "__main__":
    app()