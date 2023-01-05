---
title: "#13 React Router Dom"
header:
#   overlay_image: /assets/images/
# teaser: /assets/images/flutter.png
show_date: false
layout: single
date: 2023-01-01
classes:
  - landing
  - dark-theme
categories:
  - Nodejs, Express, MongoDB, BodyParser, PostMan, Nodemon, Bcrypt, JWT, Auth, React, React Router Dom
---

### 본 정리는 인프런 John Ahn 따라하며 배우는 노드, 리액트 시리즈 - 기본 강의를 참고하였습니다.

- **전체 흐름도**
    
    <img width="667" alt="스크린샷 2022-12-28 오후 11 21 58" src="https://user-images.githubusercontent.com/79856225/210166378-77af81fe-de50-4c5b-bbd2-89829d639358.png">

    

**하나의 페이지에서 다음페이지로 넘어가기 위한 Router 설정**

**React Router Dom 라이브러리 설치(clinet 폴더에서 입력해야 함)**

```bash
npm install react-router-dom --save
```

**App.js 파일 수정**

- 메인 페이지
- 로그인 페이지
- 회원가입 페이지

```jsx
import React from 'react';

import {
	BrowserRouter as Router,
	Switch,
	Route,
	Link
}from "react-router-dom";

import LandingPage from './components/views/LandingPage/LandingPage'
import LoginPage from './components/views/LoginPage/LoginPage'
import RegisterPage from './components/views/RegisterPage/RegisterPage'

function App(){
	return (
		<Router>
			<div>
				{
					}
				<Switch>
					<Route exact path="/" component ={LandingPage} />
					<Route exact  path="/login " component = {LoginPage} />
					<Route exact  path="/register " component = {RegisterPage } />
				</Switch>
			</div>
		</Router>		
	);
}

export default App;
```
