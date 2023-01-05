---
title: "#21 React Hooks"
header:
#   overlay_image: /assets/images/
# teaser: /assets/images/flutter.png
show_date: false
layout: single
date: 2023-01-05
classes:
  - landing
  - dark-theme
categories:
  - Nodejs, Express, MongoDB, BodyParser, PostMan, Nodemon, Bcrypt, JWT, Auth, React, React Router Dom, CORS, Proxy, Concurrently, Antd CSS, Redux, React Hooks
---

### 본 정리는 인프론 John Ahn 따라하며 배우는 노드, 리액트 시리즈 - 기본 강의를 참고하였습니다.

- **전체 흐름도**
    
    <img width="359" alt="스크린샷 2023-01-03 오후 4 02 47" src="https://user-images.githubusercontent.com/79856225/210709252-6f6ada20-d5f6-448c-b2c6-ade4c358b4ff.png">
    

### React vs React Hooks

**React Componet** 

| Class Component | Functional Component |
| --- | --- |
| Provide more features | Provide less features |
| Longer Code | Shorter Code |
| More Complex Code | Simpler Code |
| Slower Performance | Faster Performance |

<img width="676" alt="스크린샷 2023-01-03 오후 4 09 28" src="https://user-images.githubusercontent.com/79856225/210709265-97a6b509-498a-4eba-b27e-625d4782b8a2.png">

**—# 기존 리액트에서는 Functional Component는 빠르고 간단하지만 다양한 기능이 없어서 사용하지 못하는 경우가 많았는데 이를 해결하고자 React 16.8 버전부터 Hooks이라는 기능을 업데이트 했다.** 

<img width="515" alt="스크린샷 2023-01-03 오후 4 11 23" src="https://user-images.githubusercontent.com/79856225/210709276-0d5daf13-aee1-414c-8182-68ee6253dd55.png">

            (Class Component)                                              (Functional Component)
