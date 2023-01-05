---
title: "#4 BodyParser & PostMan & 회원가입 기능"
header:
#   overlay_image: /assets/images/
# teaser: /assets/images/flutter.png
show_date: false
layout: single
date: 2022-12-28
classes:
  - landing
  - dark-theme
categories:
  - Nodejs, Express, MongoDB, BodyParser, PostMan, Nodemon
---

### 본 정리는 인프런 John Ahn 따라하며 배우는 노드, 리액트 시리즈 - 기본 강의를 참고하였습니다.

<img width="513" alt="스크린샷 2022-12-27 오후 9 01 16" src="https://user-images.githubusercontent.com/79856225/209955590-4361cd62-0ab5-4dde-8fd0-1fd080e344aa.png">

- 클라이언트에서 보내주는 정보를 받기 위해서는 Body-parser 필요
- 다음 명령어를 이용하여 설치
    
    ```bash
    npm install body-parser --save
    ```
    
    <img width="299" alt="스크린샷 2022-12-27 오후 9 03 08" src="https://user-images.githubusercontent.com/79856225/209955604-8791e150-eab1-4f30-b44a-a9c0211b525c.png">
    
- 포스트맨  : http 요청을 날리고 응답을 보여주는 서비스인
- 자신의 운영체제에 맞게 포스트맨 다운로드
    
    [Postman API Platform | Sign Up for Free](https://www.postman.com/)
    
- **Register Route 생성**
    - index.js 파일에 다음 코드 추가
        
        ```jsx
        const {User} = require("./Models/User")// 미리 정의했던 모델 가져오기
        const bodyParser = require('body-parser')
        
        // 데이터 분석을 위한 추가 설정
        app.use(bodyParser.urlencoded({extended:true}));  
        app.use(bodyParser.json());
        
        app.post('/register', (req,res) =>{
            // 회원 가입할 때 필요한 정보들을 클라이언트로부터 받으면 데이터베이스에 정보 저장
            // 미리 정의했던 모델을 가져와야 함
            const user = new User(req.body);
            user.save((err, userInfo) =>{// user모델에 정보들 저장
                //만약 에러가 발생 시 json형식으로 에러와 에러메시지 전달
                if(err) return res.json({success:false, err})
                return res.status(200).json({
                    success:true
                })
            })
        })
        ```
        
        - 전체코드
            
            ```
            const express = require('express')
            const app = express()
            const port = 3000
            const mongoose = require('mongoose')
            
            const {User} = require("./Models/User")// 미리 정의했던 모델 가져오기
            const bodyParser = require('body-parser')
            
            // 데이터 분석을 위한 추가 설정
            app.use(bodyParser.urlencoded({extended:true}));  
            app.use(bodyParser.json());
            
            mongoose.set('strictQuery',true)
            mongoose.connect('mongodb+srv://Jun:zxc123@junprojcet.kzx4jm1.mongodb.net/?retryWrites=true&w=majority',
            {
                useNewUrlParser: true, useUnifiedTopology: true 
            }).then(() => console.log('Successfully connected to mongodb'))
            .catch(e => console.error(e));
            
            app.get('/', (req, res) => {
              res.send('Hello World!')
            })
            
            app.post('/register', (req,res) =>{
                // 회원 가입할 때 필요한 정보들을 클라이언트로부터 받으면 데이터베이스에 정보 저장
                // 미리 정의했던 모델을 가져와야 함
                const user = new User(req.body);
                user.save((err, userInfo) =>{// user모델에 정보들 저장
                    //만약 에러가 발생 시 json형식으로 에러와 에러메시지 전달
                    if(err) return res.json({success:false, err})
                    return res.status(200).json({
                        success:true
                    })
                })
            })
            
            app.listen(port, () => {
              console.log(`Example app listening on port ${port}`)
            })
            ```
            
            ```jsx
            const express = require('express')
            const app = express()
            const port = 3000
            const mongoose = require('mongoose')
            
            const {User} = require("Models/User")// 미리 정의했던 모델 가져오기
            const bodyParser = require('body-parser')
            
            // 데이터 분석을 위한 추가 설정
            app.use(bodyParser.urlencoded({extended:true}));  
            app.use(bodyParser.json());
            
            mongoose.connect('mongodb+srv://Jun:zxc123@junprojcet.kzx4jm1.mongodb.net/?retryWrites=true&w=majority',
            {
                useNewUrlParser: true, useUnifiedTopology: true 
            }).then(() => console.log('Successfully connected to mongodb'))
            .catch(e => console.error(e));
            
            app.get('/', (req, res) => {
              res.send('Hello World!')
            })
            
            app.post('/register', (req,res) =>{
                // 회원 가입할 때 필요한 정보들을 클라이언트로부터 받으면 데이터베이스에 정보 저장
                // 미리 정의했던 모델을 가져와야 함
                const user = new User(req.body);
                user.save((err, userInfo) =>{// user모델에 정보들 저장
                    //만약 에러가 발생 시 json형식으로 에러와 에러메시지 전달
                    if(err) return res.json({success:false, err})
                    return res.status(200).json({
                        success:true
                    })
                })
            })
            
            app.listen(port, () => {
              console.log(`Example app listening on port ${port}`)
            })
            ```
            
- 코드를 실행 후 **포스트맨에서 확인**
    
    —# Error : MongooseServerSelectionError: Could not connect to any servers in your MongoDB Atlas cluster. One common reason is that you're trying to access the database from an IP that isn't whitelisted. Make sure your current IP address is on your Atlas cluster's IP whitelist: [https://docs.atlas.mongodb.com/security-whitelist/](https://docs.atlas.mongodb.com/security-whitelist/)
    
    아이피 주소가 바뀌어서 생긴 오류로 Nerwork Access에서 현재 IP로 변경해주면 해결이 가능하다.
    
    - localhost에 Json형식으로 POST 요청 후 확인하면 아래와 같이 true가 나오면 정상적으로 요청이 완료
    
    <img width="838" alt="스크린샷 2022-12-27 오후 9 47 39" src="https://user-images.githubusercontent.com/79856225/209955613-690dbf75-f156-49e5-8fdb-28e95a43fef4.png">