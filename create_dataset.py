#학습 데이터를 만드는 코드

import cv2
import mediapipe as mp
import numpy as np
import time, os

actions = ['dog', 'cat', 'cow'] #학습시킬 단어 (각각 0,1,2에 매칭)
seq_length = 30 #하나의 데이터가 가지는 sequence길이 (30개씩 쪼개서 학습시킨다) -> LSTM 사용시 필요함
secs_for_action = 30 #각 데이터당 30초씩 학습을 시킴

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

created_time = int(time.time())
os.makedirs('dataset', exist_ok=True) #dataSet 저장 폴더 생성

while cap.isOpened():
    for idx, action in enumerate(actions):
        data = []

        ret, img = cap.read()

        img = cv2.flip(img, 1)

        cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        cv2.imshow('img', img)
        cv2.waitKey(3000) #3초동안 대기 (준비 시간)

        start_time = time.time()

        while time.time() - start_time < secs_for_action: #각 프레임을 읽어 mediaPipe에 넣는다
            ret, img = cap.read()

            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility] # 각 각도정보 및 손가락의 랜드마크가 보이는 여부를 추가

                    # Compute angles between joints
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint (각도를 구한다)
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                    v = v2 - v1 # [20, 3]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                    angle = np.degrees(angle) # Convert radian to degree

                    angle_label = np.array([angle], dtype=np.float32)
                    angle_label = np.append(angle_label, idx) #idx (=라벨)을 넣어준다.

                    d = np.concatenate([joint.flatten(), angle_label]) #100개의 행렬이 완성

                    data.append(d)

                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                break

        data = np.array(data)
        print(action, data.shape) #30초 동안 모은 데이터를 np배열로 변환
        np.save(os.path.join('dataset', f'raw_{action}_{created_time}'), data) #npy형태로 저장한다.

        # Create sequence data (30개씩 모아 시퀸스 데이터를 만든다.)
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        np.save(os.path.join('dataset', f'seq_{action}_{created_time}'), full_seq_data)

        # 즉 raw, seq 2개의 파일을 생성한다. (실제 seq파일을 사용하게됨)
    break
