---
title: "#17 Concurrently"
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
  - Nodejs, Express, MongoDB, BodyParser, PostMan, Nodemon, Bcrypt, JWT, Auth, React, React Router Dom, CORS, Proxy, Concurrently
---

### 본 정리는 인프론 John Ahn 따라하며 배우는 노드, 리액트 시리즈 - 기본 강의를 참고하였습니다.

- **전체 흐름도**
    
    <img width="663" alt="스크린샷 2022-12-30 오전 12 02 39" src="https://user-images.githubusercontent.com/79856225/210307917-4052e8ca-2986-41d6-bbee-53c08eee881b.png">

    

**Concurrently를 이용하여 프론트와 백 서버를 한 번에 켤 수있다.**

**Concurrently 라이브러리 설치 (workspace 폴더)**

```bash
npm install concurrently --save
```

**Workspace 폴더에  package.json 스크립트를 수정**

```jsx
"dev" : "concurrently \"npm run backend\" \"npm run start --prefix clinet\"" 
```

**전체 코드**

- **index.js 파일이 server 폴더로 옮겨졌으므로 그 경로에 맞게 Main경로를 수정**

```jsx
{
  "name": "boiler-plater",
  "version": "1.0.0",
  "description": "",
  "main": "./server/index.js",
  "scripts": {
    "start": "node index.js",
    "backend": "nodemon index.js",
    "test": "echo \"Error: no test specified\" && exit 1",
    "dev" : "concurrently \"npm run backend\" \"npm run start --prefix clinet\"" 
  },
  "author": "Jun",
  "license": "ISC",
  "dependencies": {
    "bcrypt": "^5.1.0",
    "body-parser": "^1.20.1",
    "concurrently": "^7.6.0",
    "cookie-parser": "^1.4.6",
    "express": "^4.18.2",
    "jsonwebtoken": "^9.0.0",
    "mongoose": "^6.8.1"
  },
  "devDependencies": {
    "nodemon": "^2.0.20"
  }
}
```

**npm run dev 명령어 실행 후 확인**