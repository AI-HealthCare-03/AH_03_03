## Food Classifier Test Platform for Trained model with PHOTOMATO
- 마지막 업데이트: 2021/02/21
- 개발환경: Ubuntu 16.04 (Windows 10) 테스트 완료
- python 환경설정: conda env (webdev) with python=3.5
- 필요 팩키지: Flask, opencv, pillow
- **최종적으로 Naver Server 에 REST API 를 Flask framework 을  이용하여 서비스함.**

- Windows 10 test: working on anaconda environment with pyTh37-pyTorch-Opencv420-office.yml
- 테스팅 파라미터는 config/config.py 를 참조.

   ```angular2html
   0. 음식데이터 라벨은 카테코리 표준코드 관리의 최종코드를 참조하여 class 로 분리함...
   1. 현재 3165종으로 구분함..
   2. YOLO 공식 홈페이지에서 제안하기로는 각 클래스당 적어도 2000개이상의 영상이 필요함.
   3. 현재 PHOTOMATO dataset 기준으로 1000번 수행 기준 2GPU 로 10시간씩 걸림..
   4. 적어도 100,000번의 수행이 경험상 필요하다고 판단되나 시간관계상 20000번을 통해 모델 취득
   5. 중간결과 분석
      13000번 이상 수행 후 쑥개떡(class_id:384), 멍게(class_id:563)의 경우
      "쑥개떡 0.78852534  - confidence: 0.7352499  - thres :  0.1" :
      "멍게 0.67948073  - confidence: 0.59680355  - thres :  0.1 " :
      계속 수행함에 따라(약 200 epoch 더 진행 후, 2시간 후).
      "쑥개떡 0.78726465  - confidence: 0.6972475  - thres :  0.1" :
      "멍게  0.7939826  - confidence: 0.69312614  - thres :  0.1 " :
      로 제대로 인식함..
      15000번: 멍게의 경우를 보면 상당히 성능이 좋아지고 있음.
          쑥개떡을 보면 아직 불안정한 것으로 판단할 수 있음
          쑥개떡: 0.55149615  - confidence: 0.4864739  - thres :  0.1
          멍게: 0.93912053  - confidence: 0.65867484

   ```
## Preparing data
```angular2html
1. data 준비
   영상의 위치는 상관이 없으나 기계적인 처리를 위해 json 파일이 같은 폴더안에 존재하는게 좋다.
   json-> yolo txt 파일 포맷 (class_id, center_x, center_y, width, height)의 tuple 로
   저장이 되어 있어야 한다. 단 좌표는 물체를 둘러쌓는 box의 중심, 가로길이, 세로길이는 반드시 전체영상의 가로와 세로로 정규화 되어야 한다.
2. data가 준비되면 training data:validation data의 비율에 따라
   train.txt 와 val.txt에 나누어 목록을 만든다. 현재의 문서는 (8:2)로 비율을 정하였다.
3. class/label/category를 나타내는 목록은
 ./yolo/data/food/food-classes.names 에 넣어져야 한다. 현재 3165개의 클래스가 들어가 있다.
```
##  Training
```angular2html
1. 데이터가 준비가 되면 ./yolo/config/food-darknet-v1.data 에 아래와 같이 훈련에 사용할 train list, validation list,
    클래스 정의, 그리고 모델을 임시 저장할 폴더 등에 대하 정의를 다음과 같이 한다. 본인의 환경에 맞게 정해 주면 된다.
    classes =3165
    train  = /workspace/food_classification/yolo/data/food/food_train_20210211.txt
    valid  = /workspace/food_classification/yolo/data/food/food_val_20210211.txt
    names = /workspace/food_classification/yolo/data/food/food-classes.names
    backup = /workspace/food_classification/yolo/data/food/weights/
2. docker를 통해 미리 만들어진 container로 들어간다. (docker 유경험자는 알겠지만 image가 없으면 자동으로 archive에서 받게 된다.)
    $ sudo docker run --gpus all -it -v ~/workspace:/workspace --ipc=host sangkny/darknet:yolov4 /bin/bash
3. docker 내에서 training을 시작한다.
    $ ./darknet detector train /workspace/food-classifier/yolo/config/food-darknet-v1.data /workspace/food-classifier/yolo/config/food-dark-yolov3-tiny_3l-v3-2.cfg /workspace/food-classifier/yolo/config/darknet53.conv.74 -gpus 0,1 2>&1 |tee /workspace/food-classifier/yolo/data/food/food-train-v3-highGPU.log
```

### Inference and WebService
```angular2html
1. 훈련된 모델에 대한 테스트는 conda 가상환경을 만들고 그안에서 실시했다.
    $ conda create --name webdev python=3.5 flask, opencv=3.4.2, pillow
    $ conda activate webdev
    위의 명령까지 정상적으로 실행이 되면 가상환경 내에 있어야 하며 다음과 유사한 프롬프트 상에 놓이게 된다.
    $ (webdev)
2. 본격적인 inference test는 food_classifier_yolo.py 로 구현이 되어 있으면
    $ python food_classifier_yolo.py 를 실행하면 된다.
    각종 parameter 조정은 ./config/config.py 를 참조하여 조절하면 된다.
3. REST API 를 Flask framework 를 통해 구현하였다.
    $ python WebAPI.py
    를 실행하면 된다. 특히, local ip(0.0.0.0, 127.0.0.1) 을 ubuntu, windows 각각 local host로 넣어주어야 한다.
    local host는 서버 본체이며, 포트(내/외부 경로 공통)를 통해 외부에서 들어오는 요청을 받아들이게 된다. 외부에서의 요청은
    공식 웹서버 ip를 통해 전달된다.
    각 web환경에 맞게 WebAPI.py 제일 마지막 main 함수에 있는 ip 와 port를 조정하여 서비스를 하면 된다.
    현재 네이버 서버는 http://xxx.xxx.xx.x:8080/uploader 로 접속을 하면 파일을 전송하라는 webpage가 뜨고
    적절한 파일을 선택 후 전송을 하여 request를 하면
    잠시후 서버에서  json의 포맷으로 탐지된 물체에 대한 정보를 나타나게 된다.
4. 이렇게 웹서버에 띄워놓은 framework(Flask)가 로그 아웃이 되어도 지속되게 하기 위해 제일 마지막에 &를 붙여
    $ python WepAPI.py &
    로 백그라운드에 daemon 같이 쓸 수 있다.
    특정 process가 죽어도 계속 써비스 하기 우해서는 no hang up의 약자인 nohup을 앞에 붙여 사용한다.
    $ hohup python WepAPI.py &
    여기에 로그를 저장하고 싶으면
    $ nohup python WebAPI.py > app.log &
    로 사용하면 된다. 하지만 화면에 출력까지 보고 싶다면,
    $ nohup python WebAPI.py 2>&1 |tee app.log &
    하면된다.
5. 웹서버의 응답은 WebAPI.py 에 정의된 것과 같이..
    (ClassID, ClassName, x, y, width, height) 를 탐지된 물체의 개수 만큼 json format 으로 넘겨주게 된다.

    ----------- 함수 정의 ------- WepAPI.py 의 food_classifier_Json(image) 은 food_classifier_yolo.py 에 있음 ---------------
    def food_classifier_Json(image):
    # do somthing
    print(args.showText)
    locations = food_classifier_pipeline(frame=image) #[(2321, 0, 0, 10, 10)] # list of (id, rect) from classification
    jsons = []
    for j,location in enumerate(locations):
        class_id, x, y, width, height =location
        res_json = {}
        res_json["ClassID"] = classes_codes[class_id] # code , class_id (training class)
        res_json["ClassName"] = classes[class_id]
        res_json["x"] = int(x)
        res_json["y"] = int(y)
        res_json["w"] = int(width)
        res_json["h"] = int(height)
        jsons.append(res_json)
    print(json.dumps(jsons,ensure_ascii=False)) # debug purpose

    return json.dumps(jsons,ensure_ascii=False)

6. 서버를 재 부팅시 위의 설정에 요구되는 사항은 CTL 등은 서버관리자에게 설정을 하게 하면 된다.
```