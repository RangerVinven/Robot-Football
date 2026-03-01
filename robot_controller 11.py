import sys
import socket
import time
import os

sys.path.append("/opt/aldebaran/lib/python2.7/site-packages")
from naoqi import ALProxy

def deep_clean(robot_ip, port):
    try:
        life = ALProxy("ALAutonomousLife", robot_ip, port)
        if life.getState() != "disabled":
            life.setState("disabled")
            time.sleep(1)
    except:
        pass

    try:
        vd = ALProxy("ALVideoDevice", robot_ip, port)
        subs = vd.getSubscribers()
        for s in subs:
            if "Streamer" in s or "Safe" in s:
                vd.unsubscribe(s)
    except:
        pass

    my_pid = os.getpid()
    os.system("ps ax | grep python | grep -v grep | grep -v %d | awk '{print $1}' | xargs kill -9 2>/dev/null" % my_pid)

def main():
    robot_ip = "127.0.0.1"
    port = 9559
    udp_port = 9000

    deep_clean(robot_ip, port)

    try:
        motion = ALProxy("ALMotion", robot_ip, port)
        motion.wakeUp()
        motion.setAngles(["HeadYaw", "HeadPitch"], [0.0, 0.0], 0.2)
        time.sleep(1.0) 
        
    except Exception as e:
        sys.exit(1)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", udp_port))
    sock.setblocking(0)
    
    head_yaw = 0.0
    head_pitch = 0.0
    cmd_vx = 0.0
    cmd_vy = 0.0
    cmd_vtheta = 0.0
    active = False

    try:
        while True:
            try:
                data, addr = sock.recvfrom(1024)
                active = True

                if data.startswith("HEAD:"):
                    parts = data.split(":")
                    pitch_change = float(parts[2])
                    head_pitch += pitch_change

                elif data.startswith("HEAD_ABS:"):
                    parts = data.split(":")
                    head_yaw = float(parts[1])
                    head_pitch = float(parts[2])

                elif data.startswith("CMD:"):
                    parts = data.split(":")
                    cmd_vx = float(parts[1])
                    cmd_vy = float(parts[2])
                    cmd_vtheta = float(parts[3])

                elif data.startswith("TURN:"):
                    direction = data.split(":")[1]
                    if direction == "LEFT":
                        cmd_vx, cmd_vy, cmd_vtheta = 0.0, 0.0, 0.2
                    elif direction == "RIGHT":
                        cmd_vx, cmd_vy, cmd_vtheta = 0.0, 0.0, -0.2

                elif data == "STOP":
                    cmd_vx = 0.0
                    cmd_vy = 0.0
                    cmd_vtheta = 0.0
                    motion.stopMove()

            except socket.error:
                pass

            if active:
                motion.moveToward(cmd_vx, cmd_vy, cmd_vtheta)
                head_yaw = max(min(head_yaw, 2.0), -2.0)
                head_pitch = max(min(head_pitch, 0.5), -0.6)
                motion.setAngles(["HeadYaw", "HeadPitch"], [head_yaw, head_pitch], 0.3)
            
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()

if __name__ == "__main__":
    main()
