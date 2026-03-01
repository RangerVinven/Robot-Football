import cv2
import numpy as np
import time
import threading
import socket
import math
from ultralytics import YOLO

class VideoStream:
    def __init__(self, url):
        self.stream = cv2.VideoCapture(url)
        self.ret, self.frame = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.stream.read()
            if not ret:
                self.stopped = True
                continue
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            return None

    def stop(self):
        self.stopped = True

def main(robot_ip):
    stream_url = f"http://{robot_ip}:8080/cam.mjpg"
    udp_port = 9000
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    vs = VideoStream(stream_url).start()
    time.sleep(1.0)
    
    model = YOLO("yolov8n.pt")
    TARGET_CLASSES = [39, 41]

    DEADZONE_X = 15
    DEADZONE_Y = 15
    STEP_SIZE = 0.04

    last_command_time = 0
    COMMAND_COOLDOWN = 0.1

    last_seen_time = time.time()
    last_seen_x = 160
    last_seen_y = 120
    search_phase = 0
    scan_timer = 0
    has_seen_ball = False

    S0 = 0
    S1 = 1
    S2 = 2
    S3 = 3
    S4 = 4
    S5 = 5
    
    state = S5
    last_state = -1
    state_time = time.time()
    ball_positions = []

    last_unaligned_time = time.time()

    while True:
        frame = vs.read()
        if frame is None: break

        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        ball_detected = False

        small = cv2.resize(frame, (160, 120))
        hsv = cv2.cvtColor(cv2.medianBlur(small, 5), cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 120, 50), (10, 255, 255)) + \
               cv2.inRange(hsv, (160, 120, 50), (180, 255, 255))
        
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        x, y, radius = 0, 0, 0

        for c in cnts:
            area = cv2.contourArea(c)
            if area < 50: continue
            
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            
            if 0.6 < circularity < 1.1:
                ((cx_c, cy_c), r_c) = cv2.minEnclosingCircle(c)
                x, y, radius = int(cx_c * 2), int(cy_c * 2), int(r_c * 2)
                
                ball_detected = True
                has_seen_ball = True
                
                last_seen_time = time.time()
                last_seen_x = x
                last_seen_y = y
                
                if search_phase != 0:
                    sock.sendto(b"CMD:0.0:0.0:0.0", (robot_ip, udp_port))
                    search_phase = 0
                    
                cv2.circle(frame, (x, y), int(radius), (0, 255, 0), 2)
                break

        current_time = time.time()

        if state != last_state:
            last_state = state
            state_time = current_time

        time_in_state = current_time - state_time

        if ball_detected:
            ball_positions.append((current_time, x, y))
            ball_positions = [p for p in ball_positions if current_time - p[0] <= 2.0]
            
            ball_speed = 0.0
            if len(ball_positions) > 5:
                dt = current_time - ball_positions[0][0]
                if dt > 0:
                    dx = x - ball_positions[0][1]
                    dy = y - ball_positions[0][2]
                    ball_speed = math.hypot(dx, dy) / dt

            vtheta = 0.0
            is_aligned = False
            
            time_since_aligned = current_time - last_unaligned_time
            base_deadzone = max(DEADZONE_X, int(radius * 1.5))
            current_deadzone = base_deadzone * 2 if state == S0 and time_since_aligned > 1.5 else base_deadzone
            is_very_close = (radius > 25 and y > 120)

            if x < (cx - current_deadzone) and not is_very_close:
                vtheta = 0.2
            elif x > (cx + current_deadzone) and not is_very_close:
                vtheta = -0.2
            else:
                is_aligned = True

            if not is_aligned:
                last_unaligned_time = current_time
                time_since_aligned = 0.0

            vx = 0.0
            vy = 0.0

            if time_in_state < 1.0:
                vx, vy, vtheta = 0.0, 0.0, 0.0
            else:
                if state == S4:
                    results = model(frame, verbose=False, classes=TARGET_CLASSES)
                    target_detected = False
                    target_x = 0
                    target_y = 0
                    max_area = 0
                    
                    for r in results:
                        for box in r.boxes:
                            x1, y1, x2, y2 = box.xyxy[0]
                            area = (x2 - x1) * (y2 - y1)
                            if area > max_area:
                                max_area = area
                                target_x = int((x1 + x2) / 2)
                                target_y = int((y1 + y2) / 2)
                                target_detected = True
                    
                    if target_detected:
                        cv2.circle(frame, (target_x, target_y), 15, (255, 0, 0), 3)
                        target_error = target_x - x
                        
                        if radius < 20:
                            vx = 0.6
                            vy = 0.0
                        else:
                            vx = 0.0
                            if target_error < -60:
                                vy = -0.5
                            elif target_error < -30:
                                vy = -0.2
                            elif target_error > 60:
                                vy = 0.5
                            elif target_error > 30:
                                vy = 0.2
                            else:
                                vy = 0.0
                                state = S0
                    else:
                        if radius < 20:
                            vx = 0.6
                            vy = 0.0
                        else:
                            vx = 0.0
                            vy = 0.5 
                        
                elif state == S0:
                    if not is_aligned and time_since_aligned < 0.5:
                         vx = 0.5 
                    elif not is_aligned:
                         state = S2
                    else:
                        if time_since_aligned > 1.5 or is_very_close:
                            vx = 1.0
                        else:
                            vx = 0.0
                        
                        if radius > 20 and y > 140:
                            state = S1

                elif state == S1:
                    vx, vtheta = 1.0, 0.0
                    if time_in_state > 4.0:
                        state = S2
                        ball_positions = []

                elif state == S2:
                    vx = 0.0
                    if time_in_state > 2.0:
                        if ball_speed < 15.0:
                            state = S5

                elif state == S3:
                    if time_in_state < 3.0:
                        vx, vtheta = -0.5, 0.0
                    else:
                        vx = 0.0
                        if is_aligned:
                            state = S5

                elif state == S5:
                    results = model(frame, verbose=False, classes=TARGET_CLASSES)
                    target_detected = False
                    for r in results:
                        if len(r.boxes) > 0:
                            target_detected = True
                            break
                            
                    if target_detected:
                        vx, vy, vtheta = 0.0, 0.0, 0.0
                        state = S4
                    else:
                        vx, vy = 0.0, 0.0
                        vtheta = 0.3
                        scan_timer += 0.1
                        pitch_scan = 0.0 + (math.sin(scan_timer) * 0.3)
                        sock.sendto(f"HEAD_ABS:0.0:{pitch_scan}".encode(), (robot_ip, udp_port))

            if time.time() - last_command_time > COMMAND_COOLDOWN:
                sock.sendto(f"CMD:{vx}:{vy}:{vtheta}".encode(), (robot_ip, udp_port))
                if time_in_state >= 1.0:
                    pitch_move = 0.0
                    if y < (cy - DEADZONE_Y): pitch_move = -STEP_SIZE
                    elif y > (cy + DEADZONE_Y): pitch_move = STEP_SIZE
                    if pitch_move != 0:
                        sock.sendto(f"HEAD:0:{pitch_move}".encode(), (robot_ip, udp_port))
                
                last_command_time = time.time()

        elif has_seen_ball:
            time_lost = current_time - last_seen_time
            if state == S1:
                if time_in_state > 4.0:
                    state = S2
                    state_time = current_time
                elif time.time() - last_command_time > COMMAND_COOLDOWN:
                    vx = 1.0 if time_in_state >= 1.0 else 0.0
                    sock.sendto(f"CMD:{vx}:0.0:0.0".encode(), (robot_ip, udp_port))
                    last_command_time = time.time()
            elif state == S0 and last_seen_y > 140 and time_lost < 0.5:
                state = S1
                state_time = current_time
            elif state == S3 and time_in_state < 3.0:
                if time.time() - last_command_time > COMMAND_COOLDOWN:
                    vx = -0.5 if time_in_state >= 1.0 else 0.0
                    sock.sendto(f"CMD:{vx}:0.0:0.0".encode(), (robot_ip, udp_port))
                    last_command_time = time.time()
            else:
                if 0.5 < time_lost < 1.5:
                    search_phase = 1
                    if time.time() - last_command_time > COMMAND_COOLDOWN:
                        vtheta = 0.2 if last_seen_x < cx else -0.2
                        sock.sendto(f"CMD:0.0:0.0:{vtheta}".encode(), (robot_ip, udp_port))
                        pitch_move = STEP_SIZE if last_seen_y > cy else -STEP_SIZE
                        sock.sendto(f"HEAD:0:{pitch_move}".encode(), (robot_ip, udp_port))
                        last_command_time = time.time()
                elif time_lost >= 1.5:
                    search_phase = 2
                    if time.time() - last_command_time > COMMAND_COOLDOWN:
                        vtheta = 0.2 if last_seen_x < cx else -0.2
                        sock.sendto(f"CMD:0.0:0.0:{vtheta}".encode(), (robot_ip, udp_port))
                        scan_timer += 0.2
                        pitch_scan = 0.0 + (math.sin(scan_timer) * 0.4)
                        sock.sendto(f"HEAD_ABS:0.0:{pitch_scan}".encode(), (robot_ip, udp_port))
                        last_command_time = time.time()

        cv2.rectangle(frame, (cx - DEADZONE_X, cy - DEADZONE_Y), (cx + DEADZONE_X, cy + DEADZONE_Y), (255, 255, 255), 1)
        cv2.line(frame, (cx, cy-10), (cx, cy+10), (255, 255, 255), 1)
        cv2.line(frame, (cx-10, cy), (cx+10, cy), (255, 255, 255), 1)

        cv2.putText(frame, f"S: {state}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("V", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    IP = "10.85.8.120"
    main(IP)
