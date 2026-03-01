import sys
import time
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import socket
from PIL import Image
from io import BytesIO

sys.path.append("/opt/aldebaran/lib/python2.7/site-packages")
from naoqi import ALProxy
import vision_definitions

video_proxy = None
video_client = None

class CamHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path.endswith('.mjpg'):
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            while True:
                try:
                    al_img = video_proxy.getImageRemote(video_client)
                    if al_img:
                        width, height = al_img[0], al_img[1]
                        img = Image.frombytes("RGB", (width, height), al_img[6])
                        
                        tmp_file = BytesIO()
                        img.save(tmp_file, format='JPEG')
                        
                        self.wfile.write("--jpgboundary\r\n")
                        self.send_header('Content-type', 'image/jpeg')
                        self.send_header('Content-length', str(tmp_file.tell()))
                        self.end_headers()
                        self.wfile.write(tmp_file.getvalue())
                        self.wfile.write("\r\n")
                        
                        time.sleep(0.1)
                except Exception as e:
                    break
            return

def main():
    global video_proxy, video_client
    robot_ip = "127.0.0.1"
    port = 9559
    
    try:
        video_proxy = ALProxy("ALVideoDevice", robot_ip, port)
        subs = video_proxy.getSubscribers()
        for s in subs:
            if "MJPEG_Streamer" in s:
                video_proxy.unsubscribe(s)

        unique_name = "MJPEG_Streamer_" + str(int(time.time()))
        video_client = video_proxy.subscribeCamera(unique_name, 1, vision_definitions.kQVGA, 11, 10)
        
        server = HTTPServer(('', 8080), CamHandler)
        server.serve_forever()
        
    except KeyboardInterrupt:
        pass
    finally:
        if video_proxy and video_client:
            video_proxy.unsubscribe(video_client)

if __name__ == "__main__":
    main()
