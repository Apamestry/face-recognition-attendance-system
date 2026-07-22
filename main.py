import os
import sys
import time
import warnings
from datetime import datetime
import csv
import cv2
import numpy as np

# Suppress legacy pkg_resources warnings (Python 3.12+ / Python 3.14 compatibility)
warnings.filterwarnings("ignore", category=UserWarning, module="face_recognition_models")

import face_recognition

PHOTOS_DIR = "photos"
ATTENDANCE_FILE = "attendance.csv"


def ensure_environment():
    """Ensure required directory and CSV file exist."""
    if not os.path.exists(PHOTOS_DIR):
        os.makedirs(PHOTOS_DIR)
    
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])


def load_known_faces():
    """Load face encodings dynamically from the photos directory."""
    ensure_environment()
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(PHOTOS_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            name = os.path.splitext(filename)[0]
            filepath = os.path.join(PHOTOS_DIR, filename)
            try:
                image = face_recognition.load_image_file(filepath)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(name)
                else:
                    print(f"⚠️ Warning: No face found in {filename}")
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")

    return known_face_encodings, known_face_names


def mark_attendance(name):
    """Record attendance into CSV if not already logged today."""
    ensure_environment()
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")

    already_marked = False
    with open(ATTENDANCE_FILE, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2 and row[0] == name and row[1] == current_date:
                already_marked = True
                break

    if not already_marked:
        with open(ATTENDANCE_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([name, current_date, current_time])
        print(f"✅ Marked attendance for: {name} at {current_time}")
        return True

    return False


def add_new_member_webcam():
    """Capture a new member's face straight from the webcam."""
    ensure_environment()
    print("\n--- Add New Member ---")
    member_name = input("Enter member full name (or press Enter to cancel): ").strip()
    if not member_name:
        print("Cancelled.")
        return

    safe_filename = "".join(c for c in member_name if c.isalnum() or c in (" ", "_")).rstrip()
    filepath = os.path.join(PHOTOS_DIR, f"{safe_filename}.jpg")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot access camera. Check macOS camera permissions!")
        return

    print("\nLook at the camera.")
    print("Press [SPACE] or 's' to Capture photo | Press 'q' to Cancel\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video feed.")
            break

        preview_frame = frame.copy()
        
        cv2.putText(preview_frame, f"Registering: {member_name}", (20, 40), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(preview_frame, "Press SPACE to Save | Press 'q' to Quit", (20, 80), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Register New Member", preview_frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord(' '), ord('s')):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_recognition.face_locations(rgb_frame)

            if len(faces) == 0:
                print("⚠️ No face detected! Position yourself clearly in front of the camera and try again.")
            else:
                cv2.imwrite(filepath, frame)
                print(f"✅ Successfully registered and saved: '{member_name}' ({filepath})")
                break
        elif key == ord('q'):
            print("Capture cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()


def delete_member():
    """Delete a registered member by removing their image file."""
    ensure_environment()
    files = [f for f in os.listdir(PHOTOS_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not files:
        print("\n❌ No members available to delete.")
        return

    print("\n================ DELETE A MEMBER ================")
    member_map = {}
    for idx, filename in enumerate(files, start=1):
        name = os.path.splitext(filename)[0]
        member_map[idx] = (name, filename)
        print(f"  [{idx}] {name}")
    print("  [0] Cancel")
    print("==================================================")

    choice = input("Enter the number of the member to delete: ").strip()
    if not choice.isdigit():
        print("❌ Invalid input.")
        return

    idx = int(choice)
    if idx == 0:
        print("Cancelled.")
        return

    if idx in member_map:
        name, filename = member_map[idx]
        filepath = os.path.join(PHOTOS_DIR, filename)
        try:
            os.remove(filepath)
            print(f"✅ Successfully deleted member: '{name}'")
        except Exception as e:
            print(f"❌ Error deleting file: {e}")
    else:
        print("❌ Invalid selection.")


def view_members():
    """List all registered members."""
    ensure_environment()
    files = [os.path.splitext(f)[0] for f in os.listdir(PHOTOS_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
    print("\n================ REGISTERED MEMBERS ================")
    if not files:
        print("No members registered yet.")
    else:
        for idx, name in enumerate(files, start=1):
            print(f"  {idx}. {name}")
    print("====================================================\n")


def view_attendance_logs():
    """Display current attendance records from CSV."""
    ensure_environment()
    print("\n================ ATTENDANCE LOGS ================")
    if not os.path.exists(ATTENDANCE_FILE):
        print("No logs available.")
        return

    with open(ATTENDANCE_FILE, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)

        if len(rows) <= 1:
            print("No attendance recorded yet.")
        else:
            for row in rows:
                if len(row) >= 3:
                    print(f"  {row[0]:<20} | Date: {row[1]} | Time: {row[2]}")
    print("=================================================\n")


def start_attendance_system():
    """Main live face recognition attendance engine."""
    print("\nLoading registered face encodings...")
    known_encodings, known_names = load_known_faces()
    print(f"Loaded {len(known_names)} registered face(s).")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera failed to open. Check macOS camera permissions.")
        return

    print("\nStarting camera feed... Press 'q' on video frame or Ctrl+C in terminal to return to menu.\n")

    notification_msg = ""
    notification_end_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture video.")
                break

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_names[best_match_index]
                        
                        is_new = mark_attendance(name)
                        if is_new:
                            notification_msg = f"✓ ATTENDANCE MARKED: {name}"
                            notification_end_time = time.time() + 3.0

                top, right, bottom, left = [coord * 4 for coord in face_location]
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            if time.time() < notification_end_time and notification_msg:
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 180, 0), cv2.FILLED)
                cv2.putText(frame, notification_msg, (20, 40), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)

            cv2.imshow("Face Recognition Attendance System", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopping attendance system...")

    finally:
        cap.release()
        cv2.destroyAllWindows()


def main_menu():
    """Main CLI Menu interface."""
    ensure_environment()

    while True:
        print("\n==================================================")
        print("   FACE RECOGNITION ATTENDANCE SYSTEM")
        print("==================================================")
        print("  [1] Start Attendance Tracker")
        print("  [2] Add New Member (Webcam Capture)")
        print("  [3] Delete Registered Member")
        print("  [4] View All Registered Members")
        print("  [5] View Attendance Logs")
        print("  [6] Exit")
        print("==================================================")

        choice = input("Select an option (1-6): ").strip()

        if choice == "1":
            start_attendance_system()
        elif choice == "2":
            add_new_member_webcam()
        elif choice == "3":
            delete_member()
        elif choice == "4":
            view_members()
        elif choice == "5":
            view_attendance_logs()
        elif choice == "6":
            print("\nExiting application. Goodbye!")
            sys.exit(0)
        else:
            print("❌ Invalid option. Please select a number between 1 and 6.")


if __name__ == "__main__":
    main_menu()