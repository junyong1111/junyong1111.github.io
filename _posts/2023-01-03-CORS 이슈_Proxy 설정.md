---
title: "#15 CORS 이슈, Proxy 설정"
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
  - Nodejs, Express, MongoDB, BodyParser, PostMan, Nodemon, Bcrypt, JWT, Auth, React, React Router Dom, CORS, Proxy
---

### 본 정리는 인프런 John Ahn 따라하며 배우는 노드, 리액트 시리즈 - 기본 강의를 참고하였습니다.

- **전체 흐름도**
    
    <img width="465" alt="스크린샷 2022-12-29 오후 11 31 20" src="https://user-images.githubusercontent.com/79856225/210307349-94687fdf-2575-4fc1-af13-12b176cdf37e.png">
    

**서로의 서버 포트번호가 달라서 에러가 발생**

**CORS 정책이란 ?**

- **Cross  Origin  Resource Sharing**
    
    **서로 다른 웹사이트에서 다른 도메인끼리 소통하려면 제한이 걸림**
    
- **다양한 해결방법이 존재**
    - **Proxy 설정으로 해결**

**Proxy 설정**

- **다음 명령어로 Proxy 라이브러리 다운로드(clinet 폴더)**
    
    ```bash
    npm install http-proxy-middleware --save
    ```
    
- **src/setupProxy.js 파일 생성 후 다음 코드 추가**
    - **target 포트번호는 자신의 포트번호와 맞춰야 함**
    - **기존 서버 3000 포트번호에서 포트번호를 5050으로 바꿨음**
    
    ```jsx
    const { createProxyMiddleware } = require('http-proxy-middleware');
    
    module.exports = function(app) {
      app.use(
        '/api',
        createProxyMiddleware({
          target: 'http://localhost:5050',
          changeOrigin: true,
        })
      );
    };
    ```
    
- **2개의 터미널을 이용하여 server와 client 서버 실행**
    - server 디렉토리에서 npm run start
    - client 디렉토리에서 npm run start
    
    —# 다음과 같은 에러발생 시
    
    ```bash
    Compiled with problems:X
    
    ERROR in ./node_modules/body-parser/lib/read.js 19:11-26
    
    Module not found: Error: Can't resolve 'zlib' in '/Users/dak_kiwon/Jun/boiler-plater/clinet/node_modules/body-parser/lib'
    ```
    
    <img width="1004" alt="스크린샷 2022-12-29 오후 11 49 30" src="https://user-images.githubusercontent.com/79856225/210307355-a1426628-3268-461c-84f7-9ee677723b12.png">
    
    <img width="632" alt="스크린샷 2022-12-29 오후 11 49 54" src="https://user-images.githubusercontent.com/79856225/210307357-12e15e5b-8c59-41ad-9214-ff0980a6bef1.png">

    
    **해당 오류가 나는 페이지에서 express 부분을 삭제하면 정상작동 한다!**