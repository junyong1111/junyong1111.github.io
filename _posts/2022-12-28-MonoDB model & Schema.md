---
title: "#3 MonoDB model & Schema"
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
  - Nodejs, Express, MongoDB
---

### 본 정리는 인프론 John Ahn 따라하며 배우는 노드, 리액트 시리즈 - 기본 강의를 참고하였습니다.

<img width="324" alt="스크린샷 2022-12-28 오후 8 29 36" src="https://user-images.githubusercontent.com/79856225/209817665-4dfbd0a4-d301-40c6-8515-47b09dd7cb62.png">

**Model**

- 스키마를 감싸주는 역할

**Schema**

- 하나하나의 정보들을 지정

**Models 폴더 생성**

- User.js 파일 생성 후 코드 입력
    
    ```jsx
    const mongoose = require('mongoose')
    
    const userSchema = mongoose.Schema({
        name:{
            type : String,
            maxlength : 50,
        },
        email:{
            type : String,
            trim : true, // space를 없애주는 역할
            unique :1  // 똑같은 이메일 사용금지
        },
        password:{
            type : String,
            minlength :5,
        },
        lastname:{
            type : String,
            maxlength : 50,
        },
        role:{ //관리자 또는 일반이 설정 기본은 일반
            type : Number,
            default : 0
        },
        image: String,
        token:{ //유효성 관리를 위한 토큰
            type:String,
        },
        tokenExp:{ //토큰의 유효기간
            type:Number,
        },
    })
    
    const User = mongoose.model('User', userSchema) //스키마를 모델로 감싸줌
    module.exports = {User} //다른곳에서 사용할 수 있게 하기위해
    ```