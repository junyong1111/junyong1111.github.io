---
title: "#18 Antd CSS Framwrok"
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
  - Nodejs, Express, MongoDB, BodyParser, PostMan, Nodemon, Bcrypt, JWT, Auth, React, React Router Dom, CORS, Proxy, Concurrently, Antd CSS
---

### 본 정리는 인프론 John Ahn 따라하며 배우는 노드, 리액트 시리즈 - 기본 강의를 참고하였습니다.

- **전체 흐름도**
    
    <img width="496" alt="스크린샷 2022-12-30 오전 12 16 27" src="https://user-images.githubusercontent.com/79856225/210308152-38a0925b-0774-486b-875e-126d608e86ef.png">

    

리액트에서 가장 유명한 프레임워크종류 중 **Ant Design 사용**

1. Material UI
2. React Bootstrap
3. Semantic UI
4. **Ant Design** 
5. Materialize

[https://ant.design/](https://ant.design/)

**Ant Design**

- 사이즈가 큼
- 사용이 편리하고 디자인이 깔끔함
- 배울 때 어려움

**Ant Design 설치 (client 폴더에서 설치)**

```bash
npm install antd --save
```

**clinet/src/index.js 파일에 다음 코드 추가**

```jsx
import 'antd/dist/antd.css';
```