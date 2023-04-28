---
title: "프로그래머스 레벨별 알고리즘"
header:
#   overlay_image: /assets/images/
# teaser: /assets/images/flutter.png
show_date: false
layout: single
date: 2023-04-28
classes:
  - landing
  - dark-theme
categories:
  - Algorithm, Programmers, Level1
---

### LEVEL_1

<details>
<summary> Level1 </summary>
<div markdown="1">   

### 1. 추억 점수
- 문제 설명 :  그리워하는 사람의 이름을 담은 문자열 배열 name, 각 사람별 그리움 점수를 담은 정수 배열 yearning, 각 사진에 찍힌 인물의 이름을 담은 이차원 문자열 배열 photo가 매개변수로 주어질 때, 사진들의 추억 점수를 photo에 주어진 순서대로 배열에 담아 return하는 solution 함수를 완성하시오.

- 해결 방법 : 각각의 사람당 그리움 점수를 map 자료구조로 담아준 후 해당 사진에서 등장한 인물들을 map구조에서 seach한 후 값들을 총합해주는 식으로 해결하였다.

```c++
#include <string>
#include <vector>
#include <map>

using namespace std;

vector<int> solution(vector<string> name, vector<int> yearning, vector<vector<string>> photo) {
    vector<int> answer;
    map<string, int>m;
    
    for(int i=0; i<name.size(); i++)
    {
        m.insert(pair(name[i], yearning[i]));
    }
    
    for(int i= 0; i< photo.size(); i++)
    {
        int sum = 0;
        for(int j=0; j<photo[i].size(); j++){
            sum += m[photo[i][j]];
        }
        answer.push_back(sum);
    }
    return answer;
}

```


### 2. 음양 더하기
- 문제 설명 : 어떤 정들이 있습니다. 이 정수들의 절댓값을 차례대로 담은 정수 배열 absolutes와 이 정수들의 부호를 차례대로 담은 불리언 배열 signs가 매개변수로 주어집니다. 실제 정수들의 합을 구하여 return 하도록 solution 함수를 완성하시오.

- 해결 방법 : 단순하게 flase값에는 -를 곱해주고 true값은 그대로 더해주는식으로 해결이 가능하다.

```c++
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>

using namespace std;

int solution(vector<int> absolutes, vector<bool> signs) {
    int answer = 0;
    
    for(int i=0; i<absolutes.size(); i++)
    {
        if(signs[i] == true)
        {
            answer += absolutes[i];
        }
        else
            answer += absolutes[i] * -1;
    }
    
    return answer;
}
```

### 3. 없는 숫자 더하기
- 문제 설명 : 0부터 9까지의 숫자 중 일부가 들어있는 정수 배열 numbers가 매개변수로 주어집니다. numbers에서 찾을 수 없는 0부터 9까지의 숫자를 모두 찾아 더한 수를 return 하도록 solution 함수를 완성하시오.

- 해결 방법 : 기존 numbers 배열을 정렬 후 0~9까지 있는 vector와 비교하여 만약 vector에는 있지만 numbers에 없다면 정답값을 더해주는 방식으로 해결하였으며 효율성은 고려하지 않아도 되는 문제였으므로 쉽게 해결 가능했다.

```c++
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;

int solution(vector<int> numbers) {
    sort(numbers.begin(), numbers.end());
    vector<int>v(10, 0);
    bool check = false;
    int answer = 0;
    for(int i=0; i<v.size(); i++)
        v[i] = i;
    for(int i=0; i<v.size(); i++)
    {
        check =false;
        for(int j=0; j<numbers.size(); j++)
        {
            if(v[i] == numbers[j])
            {
                check = true;
                break;
            }
            check = false;
        }
        if(check==false)
            answer += v[i];
    }
    return answer;
}
```

### 4. 약수의 개수와 덧셈
- 문제 설명 : 두 정수 left와 right가 매개변수로 주어집니다. left부터 right까지의 모든 수들 중에서, 약수의 개수가 짝수인 수는 더하고, 약수의 개수가 홀수인 수는 뺀 수를 return 하도록 solution 함수를 완성하시오

- 해결 방법 : 해당 문제는 시간복잡도를 고려할 필요 없이 해당 수의 약수의 개수를 구한 후 약수의 개수의 홀짝에 맞춰 계산을 해주면 쉽게 해결이 가능한 문제이다.

```c++
#include <string>
#include <vector>

using namespace std;

int count(int n)
{
    int answer = 0;
    for(int i=1; i<=n; i++)
    {
        if(n%i ==0)
            answer++;
    }
    return answer;
}

int solution(int left, int right) {
    int answer = 0;
    int cnt = 0;
    
    for(int i=left; i<= right; i++)
    {
        cnt = count(i);
        if(cnt % 2==0)
            answer += i;
        else
            answer -= i;
    }
    return answer;
}
```

### 5. 예산
- 문제 설명 : 부서별로 신청한 금액이 들어있는 배열 d와 예산 budget이 매개변수로 주어질 때, 최대 몇 개의 부서에 물품을 지원할 수 있는지 return 하도록 solution 함수를 완성해주세요.

- 해결 방법 : 해당 문제는 예산 Vector를 정렬하여 현재 값들을 더해서 예산을 초과하면 멈추면 된다.

```c++
#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <string>
#include <vector>

using namespace std;

int solution(vector<int> d, int budget) {
    int answer = 0;
    sort(d.begin(), d.end());
    
    int sum = 0;
    for(int i=0; i<d.size(); i++)
    {
        if(sum + d[i] <= budget){
            sum += d[i];
            answer++;
        }
        else
            break;
    }
    
    
    return answer;
}
```

### 6. 삼총사
- 문제 설명 : 한국중학교 학생들의 번호를 나타내는 정수 배열 number가 매개변수로 주어질 때, 학생들 중 삼총사를 만들 수 있는 방법의 수를 return 하도록 solution 함수를 완성하세요.

- 해결 방법 : 브루트포스 방식으로 해결해도 시간초과가 나지 않는다 따라서 3중포문을 이용하여 하나씩 비교하여 총합이 0이되면 방법의 수를 1개씩 늘려주는 방식으로 해결이 가능하다.

```c++
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

int solution(vector<int> number) {
    int answer = 0;
    int sum = 0;
    
    for(int i=0; i<number.size()-2; i++){
        sum = number[i];
        for(int j=i+1; j<number.size()-1; j++){
            sum += number[j];
            for(int k=j+1; k<number.size(); k++){
                sum += number[k];
                if(sum ==0)
                    answer++;
                sum = sum - number[k];
            }
            sum = sum - number[j];
        }
    }
    return answer;
}
```

### 7. 비밀지도
- 문제 설명 : 네오가 프로도의 비상금을 손에 넣을 수 있도록, 비밀지도의 암호를 해독하는 작업을 도와줄 프로그램을 작성하라.

- 해결 방법 : 해당 문제는 진법 변환만 할 수 있다면 어렵지 않게 해결할 수 있는 문제이다. c++ stl 라이브러리인 bitset을 사용하려고 했지만 bitset의 자리수 인자가 변수는 사용이 불가능하여 따로 진법 변환하는 함수를 만들어서 해결했다.

```c++
#include <string>
#include <vector>

using namespace std;

string dec_to_bi(int size, int n)
{
    string bi;
    for(int i=0; i<size; i++)
        bi += '0'; // 해당 자릿수 만큼 문자열 생성
    int idx = size-1;
    while(n){
        bi[idx--] = (n%2) + '0'; // 2진법으로 변환
        n = n/2;
    }
    return bi;
}

string concat(string a, string b)
{
    int size = a.size();
    string con_string ="";
    for(int i=0; i<size; i++){
        if(a[i] == '1' || b[i] =='1')
            con_string += '#'; // 둘 중 하나라도 1이라면 #
        else
            con_string += ' '; // 아니면 공백
    }
    return con_string;
}

vector<string> solution(int n, vector<int> arr1, vector<int> arr2) {
    
    vector<string> answer;
    
    for(int i=0; i<arr1.size(); i++){
        answer.push_back(concat(dec_to_bi(n, arr1[i]), dec_to_bi(n, arr2[i])));
    }

    return answer;
}
```

### 8. 숫자 문자열과 영단어
- 문제 설명 : 네오와 프로도가 숫자놀이를 하고 있습니다. 네오가 프로도에게 숫자를 건넬 때 일부 자릿수를 영단어로 바꾼 카드를 건네주면 프로도는 원래 숫자를 찾는 게임입니다.

- 해결 방법 : switch문 또는 if문으로 분기 처리를 하여 하드코딩하면 쉽게 해결가능

```c++
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;

int char_to_int(string str, int *answer, int idx)
{
    *answer = (*answer) * 10;  
    if(str[idx] == 'z'){
        *answer = (*answer) + 0;
        return 4;
    }
    else if(str[idx] == 'o'){
        *answer = (*answer) + 1;
        return 3;
    }
    else if(str[idx] == 't'){
        if(str[idx+1]== 'w'){
            *answer = (*answer) + 2;
            return 3;
        }
        else{
            *answer = (*answer) + 3;
            return 5;
        }
    }
    else if(str[idx] == 'f'){
        if(str[idx+1] == 'o'){
            *answer = (*answer) + 4;
            return 4;
        }
        else{
            *answer = (*answer) + 5;
            return 4;
        }
    }
    else if(str[idx] == 's'){
        if(str[idx+1] == 'i'){
            *answer = (*answer) + 6;
            return 3;
        }
        else{
            *answer = (*answer) + 7;
            return 5;
        }
    }
    else if(str[idx] == 'e'){
        *answer = (*answer) + 8;
        return 5;
    }
    else if(str[idx] == 'n'){
        *answer = (*answer) + 9;
        return 4;
    }
    else{
        *answer = (*answer) + str[idx] - '0';
        return 1;
    }
}

int solution(string s) {
    int answer = 0;
    int i = 0;
    while(i<s.size()){
        i = i + char_to_int(s, &answer, i);    
    }
    
    return answer;
}
```

### 9. 콜라문제
- 문제 설명 : 콜라를 받기 위해 마트에 주어야 하는 병 수 a, 빈 병 a개를 가져다 주면 마트가 주는 콜라 병 수 b, 상빈이가 가지고 있는 빈 병의 개수 n이 매개변수로 주어집니다. 상빈이가 받을 수 있는 콜라의 병 수를 return 하도록 solution 함수를 작성해주세요.

- 해결 방법 : 해당 문제는 주어진 조건을 적절하게 사용하여 식을 만드는 단순한 코드 구현문제로 약간의 수학적 지식과 코드 구현력만 있다면 쉽게 해결이 가능한 문제이다.

```c++
#include <string>
#include <vector>

using namespace std;

int solution(int a, int b, int n) {
    int answer = 0;
    while(n>=a){
        answer = answer + (n/a)*b;
        if(n%a ==0){
            n = (n/a)*b;
        }
        else{
            int remain = n%a;
            n = (n/a)*b + remain;
        }
    }
    return answer;
}
```

</div>
</details>

<!-- 
<details>
<summary>  </summary>
<div markdown="1">   

</div>
</details> -->
