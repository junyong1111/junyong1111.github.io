---
title: "1. 프로젝트 생성"
header:
#   overlay_image: /assets/images/
# teaser: /assets/images/flutter.png
show_date: false
layout: single
date: 2023-01-02
classes:
  - landing
  - dark-theme
categories:
  - SpringBoot, Java,
---

## 1. 프로젝트 생성

- **아티팩트는 프로젝트의 이름**
    
    <img width="799" alt="스크린샷 2022-12-26 오후 6 44 04" src="https://user-images.githubusercontent.com/79856225/209546527-502b6484-07a2-46c0-8977-55527aa49eea.png">
    
- 이 후 다음을 누르고 다음 라이브러리 설정은 체크하지 않고 넘어간다.

**그레이들 프로젝트를 스프링 부트로 변경하기**

- build.gradle 파일 오픈 후 다음 코드를 맨 윗줄에 추가
    
    ```java
    buildscript{
        ext{
            springBootVersion = '2.1.7.RELEASE'
        }
        repositories {
            mavenCentral()
            jcenter()
        }
        dependencies {
            classpath("org.springframework.boot:spring-boot-gradle-plugin:${springBootVersion}")
        }
    }
    ```
    
    - **ext**라는 키워드는 build.gradle에서 사용하는 전역변수를 설정하겠다는 의미로 여기서는 springBootVersion 전역변수를 생성하고 그 값을 2.1.7로 하겠다는 의미이다.
    - **repositories는** 각종 의존성(라이브러리)들을 어떤 원격 저장소에 받을 지를 정한다.
        - mavencentral → 라이브러리를 업로드하기 위해 많은 과정과 설정이 필요
        - jcenter → 위 문제를 개선하여 업로드를 간단하게 함
            
            일단 2개 모두 사용
            
    - **dependencies**는 프로젝트 개발에 필요한 의존성들을 선언하는 곳
        - 특정 버전을 명시하면 안된다
        - 이렇게 관리할 경우 각 라이브러리들의 버전 관리가 한 곳에 집중되고 버전 충돌 문제도 해결
- plugins{} 아래에 다음 코드 입력
    
    ```java
    apply plugin: 'java'
    apply plugin: 'eclipse'
    apply plugin: 'org.springframework.boot'
    apply plugin: 'io.spring.dependency-management'
    ```
    
    - 위 4개의 플러그인은 자바와 스프링 부트를 사용하기 위해서는 필수이므로 항상 추가
    
- 전체 코드
    
    ```java
    buildscript{
        ext{
            springBootVersion = '2.1.7.RELEASE'
        }
        repositories {
            mavenCentral()
            jcenter()
        }
        dependencies {
            classpath("org.springframework.boot:spring-boot-gradle-plugin:${springBootVersion}")
        }
    }
    
    apply plugin: 'java'
    apply plugin: 'eclipse'
    apply plugin: 'org.springframework.boot'
    apply plugin: 'io.spring.dependency-management'
    
    group 'com.example.project'
    version '1.0-SNAPSHOT'
    sourceCompatibility = 1.8
    
    repositories {
        mavenCentral()
    }
    
    dependencies {
        implementation('org.springframework.boot:spring-boot-starter-web')
        testImplementation('org.springframework.boot:spring-boot-starter-test')
    }
    ```
    
- 우측 상단에 변경 반영 클릭
    
    <img width="278" alt="스크린샷 2022-12-26 오후 7 17 19" src="https://user-images.githubusercontent.com/79856225/209546532-91364b86-2231-4fab-b867-c76aae1600ee.png">
    

**깃허브 연동하기**

- Command + Shift + A를 이용하여 Action 검색창 오픈
    - share project on github 검색 후 계정 연동
- Command + K 명령어를 통해 깃 터미널을 열 수 있음