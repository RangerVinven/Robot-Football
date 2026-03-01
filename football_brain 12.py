import cv2
import numpy as np
import time
import threading
import socket
import math

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

    STATE_APPROACHING = 0
    STATE_WALKING_THROUGH = 1
    STATE_TRACKING = 2
    STATE_BACKWARD = 3
    
    state = STATE_APPROACHING
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
            current_deadzone = base_deadzone * 2 if state == STATE_APPROACHING and time_since_aligned > 1.5 else base_deadzone
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
                if state == STATE_APPROACHING:
                    if not is_aligned and time_since_aligned < 0.5:
                         vx = 0.5 
                    elif not is_aligned:
                         state = STATE_TRACKING
                    else:
                        if time_since_aligned > 1.5 or is_very_close:
                            vx = 1.0
                        else:
                            vx = 0.0
                        
                        if radius > 20 and y > 140:
                            state = STATE_WALKING_THROUGH

                elif state == STATE_WALKING_THROUGH:
                    vx, vtheta = 1.0, 0.0
                    if time_in_state > 4.0:
                        state = STATE_TRACKING
                        ball_positions = []

                elif state == STATE_TRACKING:
                    vx = 0.0
                    if time_in_state > 2.0:
                        if ball_speed < 15.0:
                            if radius > 35 and y > 120:
                                state = STATE_BACKWARD
                            else:
                                state = STATE_APPROACHING

                elif state == STATE_BACKWARD:
                    if time_in_state < 3.0:
                        vx, vtheta = -0.5, 0.0
                    else:
                        vx = 0.0
                        if is_aligned:
                            state = STATE_APPROACHING

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
            if state == STATE_WALKING_THROUGH:
                if time_in_state > 4.0:
                    state = STATE_TRACKING
                    state_time = current_time
                elif time.time() - last_command_time > COMMAND_COOLDOWN:
                    vx = 1.0 if time_in_state >= 1.0 else 0.0
                    sock.sendto(f"CMD:{vx}:0.0:0.0".encode(), (robot_ip, udp_port))
                    last_command_time = time.time()
            elif state == STATE_APPROACHING and last_seen_y > 140 and time_lost < 0.5:
                state = STATE_WALKING_THROUGH
                state_time = current_time
            elif state == STATE_BACKWARD and time_in_state < 3.0:
                if time_time() - last_command_time > COMMAND_COOLDOWN:
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

        state_names = {0: "A", 1: "W", 2: "T", 3: "B"}
        cv2.putText(frame, f"S: {state_names.get(state, '?')}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("V", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    IP = "10.85.8.120"
    main(IP)
