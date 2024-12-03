import pickle
import cv2
import mediapipe as mp
import numpy as np
import nltk
from nltk.corpus import words
import time

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Ensure the NLTK words corpus is downloaded
nltk.download('words')
english_words = set(words.words())

# Open the camera
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define labels for letters
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
               10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 
               18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

# Buffers and timing variables
word_buffer = ""
current_letter = None
confirmation_start_time = None
prediction_timeout = 2  # Time in seconds to confirm a letter

# Function to save detected words to a file
def save_word_to_file(word, filename="detected_words.txt"):
    try:
        with open(filename, "a") as file:
            file.write(word)
        print(f"Word '{word}' saved to {filename}")
    except Exception as e:
        print(f"An error occurred while saving the word: {e}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        data_aux = []
        x_ = []
        y_ = []

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        if len(data_aux) < 84:
            data_aux += [0] * (84 - len(data_aux))
        elif len(data_aux) > 84:
            data_aux = data_aux[:84]

        prediction = model.predict([np.asarray(data_aux)])
        detected_letter = labels_dict[int(prediction[0])]

        # Confirm detection only if the same letter is detected for the entire timeout period
        if detected_letter == current_letter:
            if confirmation_start_time and (time.time() - confirmation_start_time >= prediction_timeout):
                word_buffer += detected_letter
                confirmation_start_time = None  # Reset confirmation timer
        else:
            current_letter = detected_letter
            confirmation_start_time = time.time()  # Start new confirmation timer

        # Check if the word buffer forms a valid word
        if word_buffer.lower() in english_words:
            save_word_to_file(word_buffer)  # Save the entire word at once
            cv2.putText(
                frame,
                f"Word: {word_buffer}",
                (10, H - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            word_buffer = ""  # Clear the buffer after saving the word

        # Display detected letter on screen
        cv2.putText(
            frame,
            f"Detected: {current_letter}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    cv2.imshow('frame', frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
