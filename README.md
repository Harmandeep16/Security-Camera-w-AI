
Home Security CCTV Monitor

YOLOv8 + OpenAI GPT + Tkinter GUI for real-time home surveillance.
Detects people, vehicles, and suspicious activity with automatic classification into NORMAL, CAUTION, and THREAT events.

Features
YOLOv8 object detection (people, cars, trucks, etc).
Behavior-based logic:
Walking past → CAUTION
Approaching house or camera → THREAT
Walking away → CAUTION
Loitering near or interacting with cars → THREAT
AI-powered event summaries via OpenAI Chat API.
Timeline logging with memory (last 10 alerts).
Automatic snapshots saved to detections/.
Tkinter GUI with live video feed, status panel, and log window.


<img width="1125" height="1071" alt="image" src="https://github.com/user-attachments/assets/43189b47-324e-45ef-b063-7a126eaa6298" />
