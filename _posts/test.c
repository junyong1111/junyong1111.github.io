#include <stdio.h>
#include <stdlib.h>

int main(){
    int arr[5] ={0};
    int avg = 0 ;
    for(int i=0; i<5; i++)
        scanf("%d", &arr[i]);
    for(int i=0;i<5;i++){
        if(arr[i] >=40){
            avg += arr[i];
        }
        else{
            avg += 40;
        }
    }//for
    avg /= 5;
    
    printf("%d ", avg);
    printf("\n");

    
    

    return 0;
}