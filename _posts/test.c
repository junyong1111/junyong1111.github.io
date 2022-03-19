#include <stdio.h>

int main(){
    int fix = 0;
    int varialble = 0;
    int labtop = 0;
    int cnt = 1;
    int BREAK_EVEN_POINT = labtop *cnt;
    scanf("%d %d %d", &fix, &varialble, &labtop);
    while(BREAK_EVEN_POINT < (fix+varialble) * cnt){
        printf("%d", BREAK_EVEN_POINT);
    }
    
    return 0;
}
