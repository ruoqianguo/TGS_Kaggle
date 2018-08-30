#include<stdlib.h>
#include<stdio.h>

int rlen(int *input, int len, int *result){

    int r = 0;  // the current run length
    int pos = 1;  // count starts from 1 per WK
    int idx = 0;
    int cnt = 0;
    for(int i = 0; i < len; i++){
        int c = input[i];
        if(c == 0){
            if(r != 0){
                result[idx++] = pos;
                result[idx++] = r;
                cnt += 1;
                pos += r;
                r = 0;
            }
            pos += 1;
        }
        else{
            r += 1;
        }
    }

    // if last run is unsaved (i.e. data ends with 1)
    if(r != 0){
        result[idx++] = pos;
        result[idx++] = r;
        cnt += 1;
        pos += r;
        r = 0;
    }
    return cnt;
}

int main(){
    int input[] = {0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0};
    int * result;
    result = (int *)malloc(sizeof(int) * 100);
    int len = rlen(input, 18, result);
    for(int i = 0; i < len; i++){
        printf("%d %d\n", result[2 * i], result[2 * i + 1]);
    }
    free(result);
    return 0;
}