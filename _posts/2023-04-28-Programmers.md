---
title: "프로그래머스 레벨별 알고리즘"
header:
#   overlay_image: /assets/images/
# teaser: /assets/images/flutter.png
show_date: false
layout: single
date: 2023-05-05
classes:
  - landing
  - dark-theme
categories:
  - Algorithm, Programmers, Level1
---

### 2018 KAKAO BLIND RECRUITMENT


<details>
<summary> 2018 KAKAO BLIND RECRUITMENT </summary>
<div markdown="1">   

### [1차]캐시
- **문제 설명**

지도개발팀에서 근무하는 제이지는 지도에서 도시 이름을 검색하면 해당 도시와 관련된 맛집 게시물들을 데이터베이스에서 읽어 보여주는 서비스를 개발하고 있다.
이 프로그램의 테스팅 업무를 담당하고 있는 어피치는 서비스를 오픈하기 전 각 로직에 대한 성능 측정을 수행하였는데, 제이지가 작성한 부분 중 데이터베이스에서 게시물을 가져오는 부분의 실행시간이 너무 오래 걸린다는 것을 알게 되었다.
어피치는 제이지에게 해당 로직을 개선하라고 닦달하기 시작하였고, 제이지는 DB 캐시를 적용하여 성능 개선을 시도하고 있지만 캐시 크기를 얼마로 해야 효율적인지 몰라 난감한 상황이다.
어피치에게 시달리는 제이지를 도와, DB 캐시를 적용할 때 캐시 크기에 따른 실행시간 측정 프로그램을 작성하시오.

- 해결방법

먼저 문자에서 대문자와 소문자를 구분하지 않으므로 모든 소문자를 대문자로 바꿔주는 upeercase 함수를 사용하였다.

이후 캐시사이즈가 0인 경우는 예외 처리를 해주었고 이후 캐시사이즈는 1이상이므로 먼저 캐시사이즈에 1개의 캐시를 넣어주고 시작하였다. 이후 모든 도시를 돌면서 캐시에 있는지 확인하고 있다면 hit 없다면 miss를 계산하였다. 이 문제에서는 cache hit이더라도 해당 캐시는 사용을 했으므로 가장 마지막으로 보내주어야 한다가 포인트이다!

```c++
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;

string to_uppercase(string s){
    for(int i=0; i<s.size(); i++){
        if(s[i] >='a' && s[i] <='z'){
            s[i] = s[i] - 32;
        }
    }
    return s;
}

int solution(int cacheSize, vector<string> cities) {
    if(cacheSize==0) // 예외처리
        return cities.size() * 5;
    
    int answer = 5;
    vector<string>cache;
    cache.push_back(to_uppercase(cities[0]));
                                                   
    int j = 0;
    for(int i=1; i<cacheSize; i++){
        for(j=0; j<cache.size(); j++){
            if(cache[j] == to_uppercase(cities[i])){ // hit
                cache.erase(cache.begin()+j);
                cache.push_back(to_uppercase(cities[i]));
                answer++;
                break;
            }
        }
        if(j==cache.size()){ // miss
            cache.push_back(to_uppercase(cities[i]));
            answer = answer + 5;
        }
    }
    
    
    for(int i=cacheSize; i<cities.size(); i++){
        for(j=0; j<cache.size(); j++){
            if(cache[j] == to_uppercase(cities[i])){ // cache hit
                cache.erase(cache.begin()+j);
                cache.push_back(to_uppercase(cities[i]));
                answer++;
                break;
            }
        }
        if(j==cache.size()){ // cache miss
            if(cache.size()==cacheSize)
                cache.erase(cache.begin());
            cache.push_back(to_uppercase(cities[i]));
            answer = answer + 5;
        }
    }
    return answer;
}
```

### [1차] 뉴스 클러스터링

- **문제 설명**

기사의 제목을 기준으로 "블라인드 전형"에 주목하는 기사와 "코딩 테스트"에 주목하는 기사로 나뉘는 걸 발견했다. 튜브는 이들을 각각 묶어서 보여주면 카카오 공채 관련 기사를 찾아보는 사용자에게 유용할 듯싶었다.
유사한 기사를 묶는 기준을 정하기 위해서 논문과 자료를 조사하던 튜브는 "자카드 유사도"라는 방법을 찾아냈다.
자카드 유사도는 집합 간의 유사도를 검사하는 여러 방법 중의 하나로 알려져 있다. 두 집합 A, B 사이의 자카드 유사도 J(A, B)는 두 집합의 교집합 크기를 두 집합의 합집합 크기로 나눈 값으로 정의된다.
예를 들어 집합 A = {1, 2, 3}, 집합 B = {2, 3, 4}라고 할 때, 교집합 A ∩ B = {2, 3}, 합집합 A ∪ B = {1, 2, 3, 4}이 되므로, 집합 A, B 사이의 자카드 유사도 J(A, B) = 2/4 = 0.5가 된다. 집합 A와 집합 B가 모두 공집합일 경우에는 나눗셈이 정의되지 않으니 따로 J(A, B) = 1로 정의한다.
자카드 유사도는 원소의 중복을 허용하는 다중집합에 대해서 확장할 수 있다. 다중집합 A는 원소 "1"을 3개 가지고 있고, 다중집합 B는 원소 "1"을 5개 가지고 있다고 하자. 이 다중집합의 교집합 A ∩ B는 원소 "1"을 min(3, 5)인 3개, 합집합 A ∪ B는 원소 "1"을 max(3, 5)인 5개 가지게 된다. 다중집합 A = {1, 1, 2, 2, 3}, 다중집합 B = {1, 2, 2, 4, 5}라고 하면, 교집합 A ∩ B = {1, 2, 2}, 합집합 A ∪ B = {1, 1, 2, 2, 3, 4, 5}가 되므로, 자카드 유사도 J(A, B) = 3/7, 약 0.42가 된다.
이를 이용하여 문자열 사이의 유사도를 계산하는데 이용할 수 있다. 문자열 "FRANCE"와 "FRENCH"가 주어졌을 때, 이를 두 글자씩 끊어서 다중집합을 만들 수 있다. 각각 {FR, RA, AN, NC, CE}, {FR, RE, EN, NC, CH}가 되며, 교집합은 {FR, NC}, 합집합은 {FR, RA, AN, NC, CE, RE, EN, CH}가 되므로, 두 문자열 사이의 자카드 유사도 J("FRANCE", "FRENCH") = 2/8 = 0.25가 된다.

- 해결방법
해당 문제는 코드 구현력만 있으면 해결 할 수 있는 난이도의 문제였다. 문제에서 대소문자 구분을 하지 않으므로 모든 문자는 대문자로 맞춰주고 시작했다. 이후 각각의 단어들은 2개의 쌍으로 나눠주어야 하며 이 때 알파벳이 아닌 문자가 포함된 경우는 제외하는 식으로 문자열을 전처리를 진행했다. 그 다음으로는 간단하다 하나의 문자열 vector에서 다른 문자열 vector를 순회하면서 같은 값이 있다면 교집합에 추가하고 없다면 합집합에 추가하는 방식으로 문제를 해결했다.

```c++
#include <string>
#include <algorithm>
#include <vector>

using namespace std;

string to_uppercase(string str){
    for(int i=0; i<str.length(); i++){
        if(str[i] >= 'a' && str[i] <= 'z')
            str[i] = str[i] - 32;
    }
    return str;
}

int clustering(vector<string>&s1, vector<string>&s2)
{
    // 있으면 새로운 교집합 
    // 없으면 s1 + 합집합
    if(s1.size() < s2.size()){
        vector<string>tmp = s1;
        s1  = s2;
        s2 = tmp;
    }
    vector<string>v1;
    vector<string>v2 = s1;
    int j = 0;
    for(int i=0; i<s2.size(); i++){
        string key = s2[i];
        bool check = false;
        for(j=0; j<s1.size(); j++){
            if(s1[j] == key){ // 교집합
                s1.erase(s1.begin() + j);
                v1.push_back(key);
                check = true;
                break;
            }
        }
        if(check == false) // s1+ 합집합
            v2.push_back(key);
    }
    return ((double)v1.size() / (double)v2.size()) * 65536;;
}


vector<string> cut_str(string from){
    vector<string>s;
    
    for(int i=0; i<from.length()-1; i++){
        string temp = "";
        if(from[i] >= 'A' && from[i] <='Z'){
            if(from[i+1] >= 'A' && from[i+1] <='Z'){
                temp += from[i];
                temp += from[i+1];
                s.push_back(temp);
            }
        }
    }
    return s;
}

int solution(string str1, string str2) {
    int answer = 0;
    str1 = to_uppercase(str1);
    str2 = to_uppercase(str2);
    vector<string>s1 = cut_str(str1);
    vector<string>s2 = cut_str(str2);
    if(s1.size() == 0 && s2.size()==0)
        return 65536;
    if(s1.size() == 0 || s2.size()==0)
        return 0;
    answer = clustering(s1, s2);
    return answer;
}
```

### [1차] 다트 게임
- **문제 설명**

카카오톡 게임별의 하반기 신규 서비스로 다트 게임을 출시하기로 했다. 다트 게임은 다트판에 다트를 세 차례 던져 그 점수의 합계로 실력을 겨루는 게임으로, 모두가 간단히 즐길 수 있다.
갓 입사한 무지는 코딩 실력을 인정받아 게임의 핵심 부분인 점수 계산 로직을 맡게 되었다. 다트 게임의 점수 계산 로직은 아래와 같다.
다트 게임은 총 3번의 기회로 구성된다.
각 기회마다 얻을 수 있는 점수는 0점에서 10점까지이다.
점수와 함께 Single(S), Double(D), Triple(T) 영역이 존재하고 각 영역 당첨 시 점수에서 1제곱, 2제곱, 3제곱 (점수1 , 점수2 , 점수3 )으로 계산된다.
옵션으로 스타상(*) , 아차상(#)이 존재하며 스타상(*) 당첨 시 해당 점수와 바로 전에 얻은 점수를 각 2배로 만든다. 아차상(#) 당첨 시 해당 점수는 마이너스된다.
스타상(*)은 첫 번째 기회에서도 나올 수 있다. 이 경우 첫 번째 스타상(*)의 점수만 2배가 된다. (예제 4번 참고)
스타상(*)의 효과는 다른 스타상(*)의 효과와 중첩될 수 있다. 이 경우 중첩된 스타상(*) 점수는 4배가 된다. (예제 4번 참고)
스타상(*)의 효과는 아차상(#)의 효과와 중첩될 수 있다. 이 경우 중첩된 아차상(#)의 점수는 -2배가 된다. (예제 5번 참고)
Single(S), Double(D), Triple(T)은 점수마다 하나씩 존재한다.
스타상(*), 아차상(#)은 점수마다 둘 중 하나만 존재할 수 있으며, 존재하지 않을 수도 있다.
0~10의 정수와 문자 S, D, T, *, #로 구성된 문자열이 입력될 시 총점수를 반환하는 함수를 작성하라.


- 해결방법
해당 문제는 별다른 예외케이스는 없는 문제였지만 입력으로 주어진 String을 주어진 조건에 맞게 나누는게 조금 까다로웠다 해당 문제는 총 3개의 파트로 나눠야 한다.
1. 점수
- 보너스 문자가 나오기 전까지 값을 점수로 취급
2. 보너스
- S,D,T 중 하나의 문자가 들어오면 보너스로 취급
3. 옵션 
- *, #가 들어오면 조건에 맞게 처리
이 중 옵션은 항상 주어지는 것은 아니다. 문자열을 3개의 파트로만 나눌 수 있다면 쉽게 해결이 가능하다.

```c++
#include <string>
#include <iostream>
#include <vector>
#include <cmath>
#include <vector>

using namespace std;

// 점수 0 ~ 10

// 보너스 S,D,T
// 각 점수를 보너스 만큼 제곱 S = 1, D = 2, T = 3

// *, #
// 스타상 : 해당 점수와 바로 전에 얻은 점수 2배
// 이전 값이 없는 경우 : 그 값만 2배
// 중복 된 경우 : 중첩된 스타상 4배
// 아차상과의 중복 : 아차상의 점수는 -2배
// 아차상 : 당첨 시 해당 점수 마이너스

vector<int> sep_bonus(string dartResult){
    vector<int>bonus;
    
    for(int i=0; i<dartResult.size(); i++){
        if(dartResult[i] == 'S'){
            bonus.push_back(1);
        }
        else if(dartResult[i] == 'D')
            bonus.push_back(2);
        else if(dartResult[i] == 'T')
            bonus.push_back(3);
    }
    return bonus;
}

vector<int> sep_socre(string dartResult){
    vector<int>bonus = sep_bonus(dartResult);
    vector<char>opt(3);
    vector<int>scores;
    int idx = 0;
    int score = 0;
    int count = 0;
    
    for(int i=0; i<dartResult.size(); i++){
        score = 0;
        idx = 0;
        if(dartResult[i+idx]>= '0' && dartResult[i+idx] <='9'){
            while(dartResult[i+idx]>= '0' && dartResult[i+idx] <='9'){
                score *= 10;
                score += dartResult[i+idx] -48;
                idx++;
            }
            int temp = pow(score, bonus[count++]);
            scores.push_back(temp);
        }
        if(dartResult[i+idx] == '*')
            opt[count-1] = '*';
        else if(dartResult[i+idx] == '#')
            opt[count-1] = '#';
        i = i + idx;
    }
    
    
    for(int i=0; i<opt.size(); i++){       
        if(i==0 && opt[i] == '*'){
            // 시작 하자마자 2배 찬스인 경우 그 값만 2배
            scores[i] = scores[i]*2; 
        }
        else if(opt[i] == '*'){ // 스타상의 경우
            scores[i] = scores[i] *2; 
            scores[i-1] = scores[i-1] *2;
        }
        else if(opt[i] == '#'){ //아차상의 경우 그 값만 -
            scores[i] = scores[i] * -1;
        }
    }
    return scores;
}

void print(vector<int>&V){
    for(auto v: V)
        cout << v << " ";
    cout << "\n";
}

int solution(string dartResult) {
    int answer = 0;
    vector<int>scores = sep_socre(dartResult);
    // print(scores);
    for(auto score : scores){
        answer += score;
    }
    return answer;
}
```

</div>
</details>

 




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


