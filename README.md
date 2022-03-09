## 유레카 프로젝트 
### 자신의 프로젝트를 Build한 과정을 기술

### 1.Git hub Repository 생성 및 Remote 연동
1. jyporse.github.io Repo생성
2. Repo주소 복사 후 Clone 생성  

→ git clone https://github.com/jyporse/jyporse.github.io.git Blog  
(Blog 폴더에 Git clone 생성)
3. branch 이름을 main으로 변경  
→ git branch -M main

### 2.Personal Access Token(PAT) 생성하기 
* Git을 push 하기 위해서는 개인용 Token을 생성해야 함
1. Github → Setting → Developer settings → Personal access tokens  

우측 상단 Generate new token 클릭 → 유효기간과 토큰메모 , repo을 체크해서 Token 생성  
__토큰을 비밀번호로 사용!__

2. Git push 확인  

→ git push origin main → 비밀번호로 Token 번호 입력

### 3.1 Jekyll 설치 및 시작
1. [Jekyll 설치 주소](https://jekyllrb-ko.github.io/) 자신의 OS에 맞게 지킬 설치
2. Jekyll new . --force 로 지킬 시작

### 3.2 원하는 Jekyll 테마 찾기
1. [원하는 테마](https://github.com/vszhub/not-pure-poole) : Not_pure_poole 
2. 우측 상단 fork하기 
3. __master__ branch를 __main__ branch로 바꿔준다. 
4. 기본 branch를 main으로 바꾼 후 page setting에서 Repo name을 자신의 블로그 주소로 입력
5. 위 Ropo생성 과정에서 Clone 부분(1-2)부터 다시 한다.
6. git push origin main으로 잘 되었는지 확인한다.

### 3.3 테마 변경
1. archive.yml 파일 수정
  * 블로그를 처음 열었을 때 나오는 메뉴바 수정 
    * 기존에 있던 날짜별 업로드 메뉴를 삭제
    * 추가로 카테고리 메뉴를 생성
2. social.yml 파일 수정
  * 블로그 첫 화면 바로가기 수정
    * 메일주소 업데이트
    * Git주소 업데이트
    * 개인용 Blog 주소 업데이트
3. _config.yml 파일 수정
  * 블로그 메인화면 수정
    * title 및 설명 변경
    * 기본 정보를 개인정보로 변경
    * 댓글 기능 추가

### 4. 댓글 기능 추가
1. [Disqus](https://disqus.com/) Disqus사이트에서 계정생성
2. 생성 후 "I want to install Disqus on my site" 클릭
3. Install Instruction을 읽어본 후 Configure를 눌러 다음을 진행
4. _config.yml 파일 수정  


commnet:
  provider:    "disqus"
  disqus:
    shortname: "jyporse"

5. Admin → Installing Disqus 설정
6. 다음을 _layout/post.html에 복사


\{% if page.comments  %}
\<h2>Commnets</h2>
<div id="disqus_thread"></div>
<script>
    let PAGE_URL = "{{site.url}}{{page.url}}"
    let PAGE_IDENTIFIER = "{{page.url}}"
    var disqus_config = function () {
    this.page.url = PAGE_URL;  // Replace PAGE_URL with your page's canonical URL variable
    this.page.identifier = PAGE_IDENTIFIER; // Replace PAGE_IDENTIFIER with your page's unique identifier variable
    };
    
    (function() { // DON'T EDIT BELOW THIS LINE
    var d = document, s = d.createElement('script');
    s.src = 'https://jyporse.disqus.com/embed.js';
    s.setAttribute('data-timestamp', +new Date());
    (d.head || d.body).appendChild(s);
    })();
</script>
<noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript">comments powered by Disqus.</a></noscript>
{% endif %}

7. 댓글 기능 확인 (Git_Markdown문법 등 여러문서에 댓글기능 추가)

### 5. 블로그 업로드
1. _posts 폴더에 새로운 markdown형식 파일 생성 (YYYY-MM-DD-TITLE.md 형태)
2.   
layout: post
title: Git markdown 문법
date: 2021-11-23 21:24
last_modified_at: 2021-11-23 21:24
tags: [Git, Markdown, tutorial]
categories: [Git]
toc:  false


* tages,categories에 원하는 키워드 입력
* toc(page 내 원하는 제목으로 바로가기) 비활성화 (활성화 시 지저분 함)

3. markdown 문법에 맞게 작성 후 업로드
* git add * → git commit -m "add:문서" → git push origin main으로 확인

### 6. 선택과제 댓글 기능 추가
* Git markdown 문법등 여러 문서에 댓글 기능을 추가했습니다.