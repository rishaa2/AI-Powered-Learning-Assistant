import cv2
import os
import time

SAVE_DIR = 'realtime_dataset'
EMOTIONS = ['happy', 'sad', 'frustrated', 'neutral']
SAMPLES_PER_EMOTION = 500
IMAGE_SIZE = 60

# Create folders
for emotion in EMOTIONS:
    os.makedirs(os.path.join(SAVE_DIR, emotion), exist_ok=True)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("\nPress 0: happy | 1: sad | 2: frustrated | 3: neutral | q: quit\n")

while True:
    key = input("Enter emotion key (0-3) or q to quit: ")
    if key == 'q':
        break
    if key not in ['0', '1', '2', '3']:
        print("Invalid input. Try again.")
        continue

    emotion = EMOTIONS[int(key)]
    emotion_path = os.path.join(SAVE_DIR, emotion)
    current_count = len(os.listdir(emotion_path))

    if current_count >= SAMPLES_PER_EMOTION:
        print(f"[{emotion.upper()}] Already has 500 samples. Skipping.")
        continue

    print(f"📷 Starting capture for: {emotion.upper()}")
    time.sleep(2)  # countdown buffer

    while current_count < SAMPLES_PER_EMOTION:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))
            save_path = f"{emotion_path}/{current_count}.jpg"
            cv2.imwrite(save_path, face)
            current_count += 1
            print(f"[{emotion}] Saved {current_count}/{SAMPLES_PER_EMOTION}")

        cv2.imshow("Capturing...", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"✅ Finished capturing for {emotion.upper()}!\n")

cap.release()
cv2.destroyAllWindows()
