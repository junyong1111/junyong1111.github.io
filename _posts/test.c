#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

int main(){
    int N = 10;
    int *arr = malloc(N);
    bool check = false;
    int cnt = 1;
    for(int i=0; i<N; i++){
        scanf("%d",&arr[i]);
        arr[i] = arr[i]%42;
    }
    int ch = arr[0];
    for(int i=1; i<N; i++){
        if(ch == arr[i] && check == false){check = true; cnt ++;}
        else{

        }
    }

    for(int i=0; i<N; i++){
        printf("%d ",arr[i]);
    }printf("\n");




    free(arr);
    return 0;
}