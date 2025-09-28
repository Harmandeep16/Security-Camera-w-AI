#!/usr/bin/env python3
"""
Home Security CCTV Monitor
YOLO + OpenAI API + GUI + Logging to TXT
"""

import time, os, cv2, base64, asyncio, aiohttp, tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk
import threading, json
from ultralytics import YOLO
import torch
from dataclasses import dataclass
from typing import List, Dict, Tuple
from collections import deque

# ==========================
# CONFIG
# ==========================
OPENAI_API_KEY = "sk-proj-mjo41wBYWDbJKHKJHKJHSGgaco3zR1INwRPpdRm78m3b1Ea7nO7lz-sZHkxLlFRT3SGSGSGSG9YhrJg66lqQxrnLBl5Hk9zgtXUPSGSGSGgIDKFNMvGm-ZrDbqivIqYA"  # your API key

@dataclass
class AppConfig:
    VIDEO_PATH: str = "1.mp4"
    OUTPUT_DIR: str = "detections"
    LOG_FILE: str = "security_log.txt"
    MODEL_NAME: str = "yolov8s.pt"

    AI_MODEL: str = "gpt-4o-mini"
    AI_COOLDOWN: float = 5.0
    AI_TIMEOUT: int = 15
    AI_RETRIES: int = 2
    FPS_TARGET: int = 10

    CONFIDENCE_THRESHOLD: float = 0.4

    def __post_init__(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        if not os.path.exists(self.VIDEO_PATH):
            raise FileNotFoundError(f"Video not found: {self.VIDEO_PATH}")

CONFIG = AppConfig()

# ==========================
# AI Service
# ==========================
class HomeSecurityAIService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.session = None

    async def get_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.AI_TIMEOUT)
            )
        return self.session

    async def call_ai(self, img_b64: str, events: List[str], frame_num: int, frame_size: Tuple[int, int], memory: List[str]) -> Dict:
        """Call OpenAI Chat Completions API with image + context"""
        past_context = "\n".join(memory) if memory else "No prior alerts."
        prompt = f"""
You are a vigilant home security AI guard. Analyze residential CCTV footage.

Camera: Mounted on the house. Protect the house and nearby cars (close).

Rules:
- People walking past on sidewalks = CAUTION.
- People approaching the house or camera = THREAT.
- People walking away from house/camera = CAUTION.
- People loitering near cars for long periods = THREAT.
- People interacting with cars (touching, opening, leaning) = THREAT.
- Do not repeat the same phrase more than twice, vary wording each time.
- Use short, direct security-style logs.

Past context:
{past_context}

Events in this frame: {events if events else "No events"}
Frame size: {frame_size}

Return JSON with:
- status: "NORMAL" | "CAUTION" | "THREAT"
- story: short summary for the timeline log
"""

        payload = {
            "model": self.config.AI_MODEL,
            "messages": [
                {"role": "system", "content": "You are a vigilant home camera security assistant. The camera is on the house. Protect the house and nearby cars."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    ],
                },
            ],
            "max_tokens": 200,
            "temperature": 0.2,
        }

        api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        for attempt in range(self.config.AI_RETRIES):
            try:
                session = await self.get_session()
                async with session.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        raw = data["choices"][0]["message"]["content"].strip()
                        try:
                            return json.loads(raw)
                        except:
                            return {"status": "CAUTION", "story": raw}
                    else:
                        print(f"OpenAI API error: {resp.status} {await resp.text()}")
            except Exception as e:
                print(f"AI error (attempt {attempt+1}): {e}")
            await asyncio.sleep(1)

        return {"status": "CAUTION", "story": "AI timeout"}

# ==========================
# GUI Application
# ==========================
class HomeSecurityCCTVApp:
    def __init__(self, root):
        self.root = root
        root.title("🏠 Home Security CCTV Monitor")

        self.config = CONFIG
        self.ai_service = HomeSecurityAIService(self.config)
        self.running, self.cap, self.frame_count = False, None, 0
        self.last_ai_time = 0
        self.memory = deque(maxlen=10)

        # Tracking helpers
        self.prev_positions = {}
        self.prev_areas = {}
        self.loitering = {}

        # GUI
        self.setup_gui()

        # Model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading YOLOv8 on {device}")
        self.model = YOLO(self.config.MODEL_NAME)

        # Async loop
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self._run_loop, daemon=True).start()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    # ---------------- GUI ----------------
    def setup_gui(self):
        main = ttk.Frame(self.root, padding="10")
        main.grid(row=0, column=0, sticky="nsew")

        # Status
        status_frame = ttk.LabelFrame(main, text="Security Status", padding="5")
        status_frame.grid(row=0, column=0, sticky="ew")
        self.status_var = tk.StringVar(value="🟢 SECURE")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, font=("Arial", 12, "bold"), foreground="green")
        self.status_label.grid(row=0, column=0)

        # Video
        self.video_label = ttk.Label(main)
        self.video_label.grid(row=1, column=0, pady=5)

        # Controls
        btns = ttk.Frame(main)
        btns.grid(row=2, column=0, pady=5)
        ttk.Button(btns, text="Start", command=self.start).grid(row=0, column=0, padx=5)
        ttk.Button(btns, text="Stop", command=self.stop).grid(row=0, column=1, padx=5)

        # Log
        log_frame = ttk.LabelFrame(main, text="Event Log", padding="5")
        log_frame.grid(row=3, column=0, sticky="nsew")
        self.log_text = scrolledtext.ScrolledText(log_frame, width=120, height=20, state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")

    def log(self, msg: str, save: bool = True):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, line + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")
        print(line)
        if save:
            with open(self.config.LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def update_status(self, status: str):
        color = {"THREAT": "red", "CAUTION": "orange", "NORMAL": "green"}.get(status, "green")
        icon = {"THREAT": "🔴", "CAUTION": "🟡", "NORMAL": "🟢"}.get(status, "🟢")
        self.status_var.set(f"{icon} {status}")
        self.status_label.configure(foreground=color)

    # ---------------- Video ----------------
    def start(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(self.config.VIDEO_PATH)
        self.running = True
        self.log("🟢 SYSTEM ACTIVE — Begin monitoring. Camera mounted on the house. Protect the house and nearby cars (close).")
        self._update()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.update_status("NORMAL")
        self.log("System stopped")

    def _update(self):
        if not self.running:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.stop()
            return
        self.frame_count += 1

        results = self.model(frame, verbose=False, conf=self.config.CONFIDENCE_THRESHOLD)[0]
        h, w = frame.shape[:2]
        events = []
        new_prev = {}

        objects = []
        for box in results.boxes:
            cls = self.model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            bbox = tuple(map(int, box.xyxy[0]))
            cx, cy = (bbox[0]+bbox[2])//2, (bbox[1]+bbox[3])//2
            area = (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])

            # Draw bounding boxes
            if cls == "person":
                color = (0, 0, 255)
            elif cls in ["car", "truck"]:
                color = (255, 165, 0)
            else:
                color = (0, 255, 0)

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, f"{cls} {conf:.2f}", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            objects.append(cls)

            # Prediction rules
            if cls == "person":
                obj_id = int(box.id[0]) if box.id is not None else None
                if obj_id is not None and obj_id in self.prev_areas:
                    prev_area = self.prev_areas[obj_id]
                    growth = area / prev_area
                    prev_cy = self.prev_positions.get(obj_id, (cx, cy))[1]

                    if growth > 1.2 and cy > prev_cy + 10:
                        events.append("person_approaching_house")
                    elif growth < 0.8 and cy < prev_cy - 10:
                        events.append("person_walking_away_from_house")

                if obj_id is not None:
                    new_prev[obj_id] = area
                    self.prev_positions[obj_id] = (cx, cy)

                # Loitering check
                if obj_id:
                    if obj_id in self.loitering:
                        (lx, ly, last_time) = self.loitering[obj_id]
                        if abs(cx - lx) < 30 and abs(cy - ly) < 30:
                            if time.time() - last_time > 15:
                                events.append("person_loitering_near_cars")
                        else:
                            self.loitering[obj_id] = (cx, cy, time.time())
                    else:
                        self.loitering[obj_id] = (cx, cy, time.time())

        self.prev_areas = new_prev
        self.update_display(frame)

        if time.time() - self.last_ai_time > self.config.AI_COOLDOWN:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            img_b64 = base64.b64encode(buf).decode("ascii")
            asyncio.run_coroutine_threadsafe(self.run_ai(img_b64, events, self.frame_count, (w, h)), self.loop)
            self.last_ai_time = time.time()

        self.root.after(int(1000 / self.config.FPS_TARGET), self._update)

    def update_display(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((960, 540), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(img)
        self.video_label.configure(image=imgtk)
        self.video_label.image = imgtk

    # ---------------- AI ----------------
    async def run_ai(self, img_b64, events, frame_num, frame_size):
        result = await self.ai_service.call_ai(img_b64, events, frame_num, frame_size, list(self.memory))
        status = result.get("status", "NORMAL").upper()
        story = result.get("story", "")

        if story.startswith("```"):
            story = story.strip("` \n")
            if story.lower().startswith("json"):
                story = story[4:].strip()
            try:
                parsed = json.loads(story)
                story = parsed.get("story", story)
                status = parsed.get("status", status)
            except:
                pass

        if status in ["CAUTION", "THREAT"]:
            self.update_status(status)
            self.log(f"{status} — {story}")
            self.memory.append(f"{status} — {story}")

            # Save snapshot silently
            filename = os.path.join(self.config.OUTPUT_DIR, f"alert_{status.lower()}_{frame_num}.jpg")
            with open(filename, "wb") as f:
                f.write(base64.b64decode(img_b64))

# ==========================
# Run
# ==========================
def main():
    root = tk.Tk()
    app = HomeSecurityCCTVApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop(), root.destroy()))
    root.mainloop()

if __name__ == "__main__":
    main()
