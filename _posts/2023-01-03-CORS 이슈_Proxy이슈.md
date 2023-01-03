---
title: "Compiled with problems:X 에러 발생 시 "
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

—# 다음과 같은 에러발생 시

```bash
Compiled with problems:X

ERROR in ./node_modules/body-parser/lib/read.js 19:11-26

Module not found: Error: Can't resolve 'zlib' in '/Users/dak_kiwon/Jun/boiler-plater/clinet/node_modules/body-parser/lib'
```

<img width="1004" alt="스크린샷 2022-12-29 오후 11 49 30" src="https://user-images.githubusercontent.com/79856225/210307355-a1426628-3268-461c-84f7-9ee677723b12.png">
    
<img width="632" alt="스크린샷 2022-12-29 오후 11 49 54" src="https://user-images.githubusercontent.com/79856225/210307357-12e15e5b-8c59-41ad-9214-ff0980a6bef1.png">

    
**해당 오류가 나는 페이지에서 express 부분을 삭제하면 정상작동 한다!**