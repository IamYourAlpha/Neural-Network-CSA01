#include<stdio.h>

int main() {

    FILE *fp1, *fp2;
    int label = 0;
    int tot = 1;
    double num;
    fp1 = fopen("./iris.txt", "r");
    fp2 = fopen("./iris2.txt", "w");
    for(int i = 0; i < 150; i++){
        for(int j = 0; j < 4; j++){
          fscanf(fp1, "%lf", &num);
          fprintf(fp2, "%lf ", num);
          printf("%f\t", num);
          }
          puts("");
          fprintf(fp2, " %d\n", label);
          if ( i == 49) label++;
          if ( i == 99) label++;
        }
        fclose(fp1);
        fclose(fp2);
    return 0;
}
