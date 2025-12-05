#import imageio # 이 라이브러리는 현재 코드에서 사용되지 않으므로 주석 처리하거나 제거할 수 있습니다.
import os
import numpy as np
import matplotlib.pyplot as plt

save_path = 'vis'

with open('progress.txt', 'r') as f:
    lines = f.read().splitlines()

for i, l in enumerate(lines):
    lines[i] = l.split('\t')

if not os.path.exists(save_path):
    os.mkdir(save_path)

i_x = lines[0].index('Epoch')
lines = np.array(lines)

# --- 1. 오류 수정: np.int 대신 int 사용 ---
x = lines[1:, i_x].astype(int)
#print(x)

# --- 일반 곡선 그리기 ---
for i in range(0, lines.shape[1]):
    yname = lines[0, i]
    xname = 'steps * 1000'
    plt.xlabel(xname)
    plt.ylabel(yname)
    y = lines[1:, i]
    
    # 빈 값 처리 (앞의 값으로 채우기)
    if y[1] == '':
        for j in range(1, len(y)):
            if y[j] == '':
                y[j] = y[j-1]
    
    # --- 2. 오류 수정: np.float 대신 float 사용 ---
    plt.plot(x, y.astype(float)) 
    plt.savefig(os.path.join(save_path, yname + '.png'))
    plt.cla()


# --- 곡선 평활화 함수 정의 ---
def smooth(arr, weight=0.9): #weight는 평활도, tensorboard 기본 0.6
    last = arr[0]
    smoothed = []
    for point in arr:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

save_path = 'vis_smooth'
if not os.path.exists(save_path):
    os.mkdir(save_path)

# --- 평활화된 곡선 그리기 ---
for i in range(0, lines.shape[1]):
    yname = lines[0, i]
    xname = 'steps * 1000'
    plt.xlabel(xname)
    plt.ylabel(yname)
    
    # --- 3. 오류 수정: np.float 대신 float 사용 ---
    plt.plot(x, smooth(lines[1:, i].astype(float))) 
    plt.savefig(os.path.join(save_path, yname + '.png'))
    plt.cla()
