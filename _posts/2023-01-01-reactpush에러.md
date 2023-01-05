---
title: "#리액트 props.history.push('/') 네비게이터 에러"
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
  - Nodejs, Express, MongoDB, BodyParser, PostMan, Nodemon, Bcrypt, JWT, Auth, React, React Router Dom
---


리액트에서 페이지를 이동할 때 다음과 같은 코드를 사용하는데 페이지 이동이 이뤄지지 않는다.

```jsx
props.history.push('/')
```

버전이 올라가면서 해당 코드 대신 다음 코드를 입력하여 해결 가능

```jsx
import { useNavigate } from 'react-router-dom';

const navigate = useNavigate();
// props.history.push('/')
navigate('/')
```

useNavigate를 import한 후 prpos.history.push대신 navigate를 사용하면 페이지 오류가 해결된다