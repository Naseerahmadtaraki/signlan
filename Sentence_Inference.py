"""
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time


# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Open the camera
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define labels for letters
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4:'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
               10: 'K', 11: 'L', 12: 'M', 13: 'N', 14:'O', 15: 'P', 16: 'Q', 17: 'R', 
               18: 'S', 19: 'T', 20: 'U', 21: 'V', 22:'W', 23: 'X', 24: 'Y', 25: 'Z'}

# Buffer to hold recognized characters
sentence_buffer = ""
last_prediction_time = time.time()
prediction_timeout = 2  # seconds to hold a character before it's considered confirmed

while True:
    ret, frame = cap.read()  # Read the frame
    if not ret:
        print("Error: Could not read frame.")
        break  # Exit the loop if the frame was not captured successfully

    # Ensure frame is not None
    if frame is None:
        print("Error: Frame is None.")
        break

    H, W, _ = frame.shape  # Access the shape of the frame

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

        # Make sure data_aux has the right number of features
        if len(data_aux) < 84:
            data_aux += [0] * (84 - len(data_aux))  # Pad if needed
        elif len(data_aux) > 84:
            data_aux = data_aux[:84]  # Trim if too long

        # Make prediction
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        # Update sentence buffer
        current_time = time.time()
        if (current_time - last_prediction_time) < prediction_timeout:
            # Only add character if it's within the timeout
            if predicted_character not in sentence_buffer[-1:]:  # Avoid duplicates
                sentence_buffer += predicted_character
        else:
            # Reset the buffer if timeout has occurred
            sentence_buffer = predicted_character

        last_prediction_time = current_time

        # Draw bounding box for the most recent recognized hand
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

        # Display the entire sentence
        cv2.putText(frame, sentence_buffer, (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
"""
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
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4:'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
               10: 'K', 11: 'L', 12: 'M', 13: 'N', 14:'O', 15: 'P', 16: 'Q', 17: 'R', 
               18: 'S', 19: 'T', 20: 'U', 21: 'V', 22:'W', 23: 'X', 24: 'Y', 25: 'Z'}

# Buffer to hold recognized characters
word_buffer = ""
last_prediction_time = time.time()
prediction_timeout = 2  # seconds to hold a character before it's considered confirmed

while True:
    ret, frame = cap.read()  # Read the frame
    if not ret:
        print("Error: Could not read frame.")
        break  # Exit the loop if the frame was not captured successfully

    H, W, _ = frame.shape  # Access the shape of the frame

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    current_time = time.time()  # current time

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

        # Make sure data_aux has the right number of features
        if len(data_aux) < 84:
            data_aux += [0] * (84 - len(data_aux))  # Pad if needed
        elif len(data_aux) > 84:
            data_aux = data_aux[:84]  # Trim if too long

        # Make prediction
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        # Update sentence buffer

        if (current_time - last_prediction_time) < prediction_timeout:
            if predicted_character not in word_buffer[-1:]:
                word_buffer += predicted_character
                save_letter_to_file(predicted_character)

        else:
            # Check if the accumulated characters form a valid word
            if word_buffer.lower() in english_words:
                print("Recognized word:", word_buffer)  # Print the recognized word
                # Reset the buffer after recognizing a valid word
                word_buffer = ""
            #else:
                # Clear the buffer if the word is not valid
             #   word_buffer = ""

        last_prediction_time = current_time

        # Draw bounding box for the most recent recognized hand
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)


        # Display the entire sentence if a valid word is formed
        if word_buffer.lower() in english_words:
            cv2.putText(frame, word_buffer, (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        # Display the entire sentence
        #cv2.putText(frame, sentence_buffer, (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

def save_letter_to_file(letter, filename="detected_letters.txt"):
    """
    Save the detected letter to a text file.
    Prevents consecutive duplicates from being added.

    :param letter: The detected letter to save.
    :param filename: The name of the file to save the letter. Default is 'detected_letters.txt'.
    """
    try:
        # Read the last saved letter to prevent consecutive duplicates
        last_letter = None
        try:
            with open(filename, "r") as file:
                lines = file.readlines()
                if lines:
                    last_letter = lines[-1].strip()  # Get the last letter saved
        except FileNotFoundError:
            # File doesn't exist yet, so we create it
            pass

        # Append the new letter only if it's not a duplicate
        if letter != last_letter:
            with open(filename, "a") as file:
                file.write(letter + "\n")
            print(f"Letter '{letter}' saved to {filename}")
        else:
            print(f"Letter '{letter}' is a duplicate and was not saved.")

    except Exception as e:
        print(f"An error occurred while saving the letter: {e}")


cap.release()
cv2.destroyAllWindows()

