---
title: "#14 데이터 Flow & Axios"
header:
#   overlay_image: /assets/images/
# teaser: /assets/images/flutter.png
show_date: false
layout: single
date: 2023-01-03
classes:
  - landing
  - dark-theme
categories:
  - Nodejs, Express, MongoDB, BodyParser, PostMan, Nodemon, Bcrypt, JWT, Auth, React, React Router Dom
---

### 본 정리는 인프런 John Ahn 따라하며 배우는 노드, 리액트 시리즈 - 기본 강의를 참고하였습니다.

- **전체 흐름도**
    
    <img width="527" alt="스크린샷 2022-12-29 오후 11 14 40" src="https://user-images.githubusercontent.com/79856225/210306985-83c18785-ef6b-4e27-a42b-f4b8b76130db.png">

    

**NodeJs → Server**

**ReactJs → Clinet**

**MongoDB → DB**

**유저가 로그인(클라이언트) → 서버에서 데이터베이스 접근 → 일치여부 확인 → 유저에게 보여줌**

**이제 클라이언트가 있으니 POSTMAN이 아닌 Axios 라이브러리 를 이용**

- **Axios 라이브러리 다운로드**
    - client 폴더로 이동 후 다음 명령어 입력
    
    ```bash
    npm install axios --save
    ```
    
- **LandingPage.js 파일 코드 추가(클라이언트)**
    
    ```jsx
    import React, {useEffect} from 'react'
    import axios from 'axios'
    import { response } from 'express'
    
    function LandingPage(){
    
        useEffect(()=>{
            axios.get('/api/hello')
            .then(response => console.log(response.data))
        }, [])
    
        return (
            <div>
                LandingPage
            </div>
        )
    }
    export default LandingPage
    ```
    
- **index.js 파일 코드 추가(서버)**
    
    ```jsx
    app.get('/api/hello', (req,res) =>{
    
      res.send("안녕하세요")
    })
    ```
    
- **2개의 터미널을 이용하여 server와 client 서버 실행**
    - server 디렉토리에서 npm run start
    - client 디렉토리에서 npm run start
    
    **확인하면 에러가 뜬다 이유는 서로의 포트번호가 달라서임  따로 설정필요**