import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Screen dimensions
screen_width, screen_height = pyautogui.size()

while True:
    # Capture frame from webcam
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame horizontally for natural interaction
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark coordinates for index and middle finger tips
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Convert normalized coordinates to pixel values
            index_x, index_y = int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0])
            middle_x, middle_y = int(middle_finger_tip.x * frame.shape[1]), int(middle_finger_tip.y * frame.shape[0])

            # Calculate the distance between the index and middle finger tips
            distance = np.hypot(index_x - middle_x, index_y - middle_y)

            # Define a threshold for the click gesture
            click_threshold = 30  # Adjust this value based on your setup

            if distance < click_threshold:
                # Perform a mouse click
                pyautogui.click()
                cv2.putText(frame, 'Click', (index_x, index_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Map hand coordinates to screen coordinates
            screen_x = np.interp(index_x, [0, frame.shape[1]], [0, screen_width])
            screen_y = np.interp(index_y, [0, frame.shape[0]], [0, screen_height])

            # Move the mouse cursor
            pyautogui.moveTo(screen_x, screen_y)

    # Display the frame
    cv2.imshow('Hand Tracking', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
