from calendar import c
import cv2
import mediapipe as mp
import time
from socketIO_client import SocketIO

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
controlling = False
INFO_TEXT_X = 15
INFO_TEXT_Y = 35
both_hands = False
left_detected = False
right_detected = False
right_hand_landmarks = []
left_hand_landmarks = []
controlling_hand_message = ""
controlling_message = "Sin control"
controlling_message_color = (0,0,0)
RIGHT_WRIST_START_X = 0
RIGHT_WRIST_START_Y = 0
# RIGHT_WRIST_START_Z = 0
X = 0
Y = 0
Z = 0
ctr = 0

print("Comenzando...")
socketIO = SocketIO('localhost', 5001)
print("Conectado al servidor.")

cap = cv2.VideoCapture(1)

# with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

while True:
    ok, frame = cap.read()
    if not ok:
        continue
    height, width, channels = frame.shape
    #frame = cv2.flip(frame,1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_handedness is not None: # is not pointer to null
        for index, hand_handedness in enumerate(results.multi_handedness):
            left_right = hand_handedness.classification[0].index # 0: Left, 1:Right
            if(left_right == 1): # is the right hand?
                right_detected = True
                right_hand_landmarks = results.multi_hand_landmarks[index]
            elif(left_right == 0): # is the left hand?
                left_detected = True
                left_hand_landmarks = results.multi_hand_landmarks[index]

        if right_detected and not left_detected:
            controlling_hand_message = "DERECHA"
            mp_drawing.draw_landmarks(frame, right_hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
        elif right_detected and left_detected:
            controlling_hand_message = "DERECHA + IZQUIERDA"
            mp_drawing.draw_landmarks(frame, right_hand_landmarks, mp_hands.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, left_hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # calibration
            if not controlling:
                cv2.circle(frame, (100, 200),10,(0,0,255),11)
                LEFT_INDEX_FINGER_TIP_X = round(left_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width)
                LEFT_INDEX_FINGER_TIP_Y = round(left_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)
                cv2.circle(frame, (LEFT_INDEX_FINGER_TIP_X,LEFT_INDEX_FINGER_TIP_Y),5,(0,0,255),6)

                if LEFT_INDEX_FINGER_TIP_X>89 and LEFT_INDEX_FINGER_TIP_X<111 and LEFT_INDEX_FINGER_TIP_Y>190 and LEFT_INDEX_FINGER_TIP_Y<211:
                    controlling = True
                    controlling_message = "SI"
                    controlling_message_color = (211,120,64)
                    # Picture of Wrist incial values
                    RIGHT_WRIST_START_X = round(right_hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * width)
                    RIGHT_WRIST_START_Y = round(right_hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * height)
                    # RIGHT_WRIST_START_Z = right_hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z



        elif left_detected and not right_detected:
            controlling_hand_message = "IZQUIERDA"
            mp_drawing.draw_landmarks(frame, left_hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if controlling and right_detected:
            #if ctr == 8:
            RIGHT_WRIST_X = round(right_hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * width)
            RIGHT_WRIST_Y = round(right_hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * height)
            #RIGHT_WRIST_Z = right_hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z
            X = (RIGHT_WRIST_X - RIGHT_WRIST_START_X)/2
            Y = (RIGHT_WRIST_Y - RIGHT_WRIST_START_Y)/2
            X = 90 + X
            Y = 90 + Y
            if X < 0:
                X = 0
            elif X > 180:
                X = 180
            if Y < 0:
                Y = 0
            elif Y > 180:
                Y = 180
            
            # print("X: "+str(X))
            # print("Y: "+str(Y))
            socketIO.emit("nuevo_mensaje",str(X)+'\n')
            #ctr = 0
        
        right_detected = False
        left_detected = False


    else:
        controlling_hand_message = "NINGUNA"
        controlling_message = "NO"
        controlling_message_color = (0,0,0)
        controlling = False
    
    cv2.putText(frame, "MANO DETECTADA: "+controlling_hand_message,(INFO_TEXT_X,INFO_TEXT_Y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 0, 0),2)
    cv2.putText(frame, "CONTROLANDO ROBOT: "+controlling_message,(INFO_TEXT_X,INFO_TEXT_Y+35),cv2.FONT_HERSHEY_SIMPLEX,0.8,controlling_message_color,2)
        
    cv2.imshow("Frame", frame)
    time.sleep(0.05)
    if cv2.waitKey(5) & 0xFF == 27:
        break
    #ctr += 1


cap.release()
cv2.destroyAllWindows()
