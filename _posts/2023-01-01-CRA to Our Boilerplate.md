---
title: "#12 CRA to Our Boilerplate"
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
  - Nodejs, Express, MongoDB, BodyParser, PostMan, Nodemon, Bcrypt, JWT, Auth, React
---

### 본 정리는 인프론 John Ahn 따라하며 배우는 노드, 리액트 시리즈 - 기본 강의를 참고하였습니다.

- **전체 흐름도**
    
    <img width="240" alt="스크린샷 2022-12-28 오후 11 00 15" src="https://user-images.githubusercontent.com/79856225/210166354-2a6d4d9e-4813-4490-9f25-3af181cdbefc.png">
    

**src 하위 경로에 다음 폴더 및 파일 추가**

- **_actions 폴더 생성**
- **_reducers 폴더 생성**
- **components 폴더 생성**
    - **views 폴더 생성**
        - **LandingPage 폴더 생성 : 처음 페이지**
            - **LandingPage.js 파일 생성 후 다음 코드 입력**
            - **ES7 React 확장팩을 설치하면 rfce를 입력하여 기본 코드 자동완성 가능**
                
                <img width="1495" alt="스크린샷 2022-12-28 오후 11 17 37" src="https://user-images.githubusercontent.com/79856225/210166355-424954bc-99e3-41bf-9644-7bdf3929f4f0.png">

                
                ```jsx
                //**LandingPage
                import React from 'react'
                
                function LandingPage(){
                    return (
                        <div>
                            LandingPage
                        </div>
                    )
                }
                export default LandingPage**
                ```
                
        - **LoginPage 폴더 생성 : 로그인 페이지**
            - **LoginPage.js 파일 생성 후 다음 코드 입력**
            
            ```jsx
            import React from 'react'
            
            function LoginPage() {
              return (
                <div>LoginPage</div>
              )
            }
            
            export default LoginPage
            ```
            
        - **RegisterPage 폴더 생성 : 회원가 입 페이지**
            - **RegisterPage.js 파일 생성 후 다음 코드 입력**
            
            ```jsx
            import React from 'react'
            
            function RegisterPage() {
              return (
                <div>RegisterPage</div>
              )
            }
            
            export default RegisterPage
            ```
            
        - **NavBar 폴더 생성 : 네비게이션 바**
            - **NavBar.js 파일 생성 후 다음 코드 입력**
            
            ```jsx
            import React from 'react'
            
            function NavBar() {
              return (
                <div>NavBar</div>
              )
            }
            
            export default NavBar
            ```
            
        - **Footer 폴더 생성 : 하단 정보**
            - **Footer.js 파일 생성 후 다음 코드 입력**
            
            ```jsx
            import React from 'react'
            
            export default function Footer() {
              return (
                <div>Footer</div>
              )
            }
            ```
            
- **utils 폴더 생성**
- **hoc 폴더 생성**
- **Config.js 파일 생성**
