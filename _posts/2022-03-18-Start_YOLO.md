---
title: "YOLO(You Only Look Once)"
header:
  overlay_image: /assets/images/img1.jpeg
teaser: /assets/images/img1.jpeg
layout: splash
classes:
  - landing
  - dark-theme
# layout: posts
date: 2022-07-21
show_date: true
categories:
  - YOLO
  - Colab
---

## YOLO V4
YOLO(You Only Look Once) : 다중 객체 인식

YOLO V4모델은 You Only Look Once의 약자로써, 객체 탐지(Object detection)분야에서 많이 알려져 있으며 이미지를 한 번 보는 것으로 물체의 종류와 위치를 추측하며 이미지의 전체 맥락을 이해하므로 빠르고 정확하다

YOLO v4는 이전v3버전을 더욱 개량하여 정확도를 향상시켰다.

## 학습환경
### Google Colab 
(코랩에서 진행하는 이유)

- #### YOLO를 학습시키기 위해서는 Darknet을 사용해야하는데 설치 조건이 까다롭기 때문에 공통적인 환경을 위함 
- #### Linux 환경
- #### GPU연산 가능  
- #### python 
#### #주의점 : Colab 무료버전은 최대 런타임 시간은 12시간이므로 구글 드라이브를 통한 데이터 백업 필요


## 학습

Colab에서 학습을 하기위해서는 최초 Darknet 환경을 빌드해야한다. 

<details>
<summary> 최초 1회 Draknet 빌드 </summary>
<div markdown="1">

### 개발환경 만들기

1. 런타임 → 런타임 유형 변경 → 하드웨어 가속기(CPU) → 하드웨어 가속기(GPU)로 설정
2. 현재 Colab과 연동되어있는 구글드라이브 마운트 

```python
from google.colab import drive
drive.mount('/content/drive')
```

위 코드를 입력하면 구글드라이브에 저장되어있는 파일들을 Colab에서 사용가능

3. GPU 사용에 필요한 CUDA 설치

```python
!/usr/local/cuda/bin/nvcc --version
## 현재 CUDA버전 확인 자신한테 맞는 버전을 확인 후 NVIDIA 홈페이지에서 버전에 맞게 다운

!arch 
## 리눅스 버전확인 cuDNN을 다운받을 때 현재 자신의 Colab 리눅스 버전에 맞게 다운
```

NVIDIA 홈페이지에서 확인
https://developer.nvidia.com/rdp/cudnn-download
회원가입 후 다운로드
cuDNN : CUDA의 소프트웨어
deep neural networks를 사용하기 위해 
자신의 CUDA 버전과 맞는 버전 다운
다운받은 파일을 구글 Drive의 darknet이란 폴더를 만들고 그 안에 cuDNN 폴더를 만들어서 옮겨놓는다.
- Googole Drive -> darknet -> cuDNN 안에 다운받은 파일을 넣음
연결된 경로복사 후 아래 명령어로 압축해제

```python
!tar -xzvf drive/MyDrive/darknet/cuDNN/cudnn-11.1-linux-x64-v8.0.5.39.tgz -C /usr/local/
## 위 경로를 자신이 저장해둔 경로에 맞게 설정한 뒤 압축해제 
!chmod a+r /usr/local/cuda/include/cudnn.h

!cat /usr/local/cuda/include/cudnn.h
## 잘 설치되었는지 설치확인
```

4. 다크넷 설치

* CUDA와 C와 기본으로 하며 빠르고 쉽다. 
* DarkNet install https://pjreddie.com/darknet/install/
* Colab에서 사용하기 편하게 바꾼 코드 실행


```python
!git clone https://github.com/AlexeyAB/darknet.git
## darknet 파일이 저장되어있는 git clone
%cd darknet
## 현재 경로 이동
!ls
# Clone 내용 확인

!git checkout feature/google-colab
```

- #Compile DarkNet (매번 할 필요없이 1회만 하면 된다)
#### Makefile 수정단계

```python
%cd /content/darknet/
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
!sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
```

```python
import os
if not os.path.exists('/content/drive/MyDrive/darknet/bin'):
    os.makedirs('/content/drive/MyDrive/darknet/bin')
# 만약 bin이라는 폴더가 없다면 bin이라는 폴더를 생성하는 코드 경로는 자신의 darknet경로로 설정
```

```python
!make
## draknet 컴파일
```

```python
!cp -r ./darknet /content/drive/MyDrive/darknet/bin/darknet
## 드라이브에 복사 경로는 자신의 darknet경로로 설정
# 컴파일 과정없이 다음부터는 해당 폴더를 불러와서 파일을 실행하면 된다.
```

최초 빌드 이후 다음부터는 아래 코드를 이용하여 구글드라이브에 이미 빌드된 다크넷을 가져와서 권한설정만 해준 뒤 사용하면된다.

```python
!cp /content/drive/MyDrive/darknet/bin/darknet ./darknet
!chmod +x ./darknet
## darknet 권한설정
```

5. 다크넷 확인

```python
#download files

def imShow(path):
    import cv2
    import matplotlib.pyplot as plt
    %matplotlib inline

    img = cv2.imread(path)
    height , width = img.shape[:2]
    resized_img = cv2.resize(img, (3*width, 3*height),interpolation = cv2.INTER_CUBIC)

    fig = plt.gcf()
    fig.set_size_inches(18,10)
    plt.axis("off")
    #plt.rcParams['figure.figsize'] = [10,5]
    plt.imshow(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
    plt.show()

def upload():
    from google.colab import files
    uploaded = files.upload()
    for name, data in uploaded.items():
        with open(name, "wb") as f:
            f.write(data)
            print("saved file", name)
def download(path):
    from google.colab import files
    files.download(path)
```

### Darknet 에서 미리 학습된 데이터를 구글드라이브에서 가져와서 복사
- weights 파일
https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
구글 드라이브 -> darknet -> weights 폴더를 만들어서 yolov4.weights 파일을 넣음
- cfg 파일
https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg
구글 드라이브 -> darknet -> cfg 폴더를 만들어서 yolov4.cfg 파일을 넣음

```python
!cp -r "/content/drive/MyDrive/darknet/weights" ./weights
!cp -r "/content/drive/MyDrive/darknet/weights" ./cfg
## 위 명령어로 드라이브에 있는 2개의 폴더를 colab 로컬 폴더로 복사
```

```python
!./darknet detect cfg/yolov3.cfg weights/yolov3.weights data/dog.jpg
## cfg는 자신의 cfg파일이 있는경로
## weights는 자신의 weights파일이 있는 경로
## data는 자신이 테스트할 사진 다크넷 빌드시 자동으로 생성된다.
```

객체 탐지가 정상적으로 완료되면 predictions.jpg 파일이 생성된다 위에서 정의함 함수를 이용하여 결과를 확인

```python
imShow('predictions.jpg')
```

#### 명령어가 실행이 안된다면 높은확률로 경로 문제일 가능성이 크다. !ls 명령어로 현재 경로를 확인 후 darknet파일이 있는 폴더까지 이동 후 cfg 와weights 경로를 다시 확인후 실행해보자

</div>
</details>

##### -------------------- Darknet 빌드



## 커스텀 데이터 학습

<details>
<summary> 1. 크롤링 </summary>
<div markdown="1">

### 커스텀 데이터를 학습시키기 위해서는 자신이 원하는 데이터셋이 필요하다. 크롤링을 해서 이미지를 얻는 방법은 여러가지가 존재 한다 원하는 자신이 원하는 방법을 이용하여 학습에 필요한 이미지파일을 50~100장 정도 수집한다.
[크롤링](https://junyong1111.github.io/크롤링/selenium/크롤링셋팅/)
위 글을 참고하여 크롤링 진행
</div>
</details>

##### -------------------- 원하는 이미지 크롤링

<details>
<summary> 2. 이미지 라벨링 </summary>
<div markdown="1">

### 학습과정에서 가장 힘든 부분이라고 생각한다 정말 단순 반복 작업이고 데이터의 양이 많거나 클래스가 많은 경우 상당한 시간이 걸린다..



</div>
</details>

##### -------------------- 이미지 라벨링







<!--
<details>
<summary>  </summary>
<div markdown="1">

</div>
</details>
----------------------
-->