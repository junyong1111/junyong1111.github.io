---
title: "#20 Redux UP"
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
  - Nodejs, Express, MongoDB, BodyParser, PostMan, Nodemon, Bcrypt, JWT, Auth, React, React Router Dom, CORS, Proxy, Concurrently, Antd CSS, Redux
---

### 본 정리는 인프론 John Ahn 따라하며 배우는 노드, 리액트 시리즈 - 기본 강의를 참고하였습니다.

- **전체흐름도**
    
    <img width="674" alt="스크린샷 2023-01-03 오후 3 24 58" src="https://user-images.githubusercontent.com/79856225/210709079-1fb74246-7e4d-4fd3-951f-d1bd86ed51fe.png">

    

**4개의 의존성 설치**

- **Clinet 디렉토리에서 터미널에서 다음 명령어 실행**
    
    ```bash
    npm install redux react-redux redux-promise redux-thunk --save
    ```
    

**Redux는 Store안에서 상태를 관리 함**

- 항상 객체형식의 액션만 받는 것은 아님
    - Promise 또는 Function으로 받을 수 있음

**Redux를 더 잘 사용할 수 있게 도와주는 미들웨어**

- **Redux-promise**
    - Dispatch한테 Promise가 왔을 때 어떻게 대처해야하는지 알려줌
- **Redux - thunk**
    - 어떻게 Function 받을지를 Dispatch한테 알려줌

### **Redux를 앱에다가 연결**

- 구글 크롬 확장 [Redux Dev Tools](https://chrome.google.com/webstore/detail/redux-devtools/lmhkpmbekcpmknklioeibfkpmmfibljd?hl=ko-KR) 다운로드 및 연결
    
    [Redux DevTools](https://chrome.google.com/webstore/detail/redux-devtools/lmhkpmbekcpmknklioeibfkpmmfibljd/related?hl=ko-KR)
    
    ```jsx
    window.__REDUX_DEVTOOLS_EXTENSION__&&
    window.__REDUX_DEVTOOLS_EXTENSION__()
    ```
    
- clinet/src/index.js 파일 수정
    
    ```jsx
    import React from 'react';
    import ReactDOM from 'react-dom';
    import './index.css';
    import App from './App';
    import reportWebVitals from './reportWebVitals';
    import { Provider } from 'react-redux';
    import {applyMiddleware} from 'redux';
    import {legacy_createStore as createStore} from 'redux';
    import promiseMiddleware from 'redux-promise';
    import ReduxThunk from 'redux-thunk';
    import Reducer from './_reducers';
    
    const createStoreWithMiddleware = applyMiddleware(promiseMiddleware, ReduxThunk)(createStore)
    
    ReactDOM.render(
      <Provider
        store={createStoreWithMiddleware(Reducer,
          window.__REDUX_DEVTOOLS_EXTENSION__ &&
          window.__REDUX_DEVTOOLS_EXTENSION__())}
      >
        <App />
      </Provider>
        
      //</React.StrictMode>,
      , document.getElementById('root'));
    
    reportWebVitals();
    ```
    
- **clinet/src/_reducers/index.js 파일 생성 후 다음 코드 입력**
    
    ```jsx
    import { combineReducers} from 'redux';
    
    const rootReducer = combineReducers({
    	//user
    })
    
    export default rootReducer ;
    ```