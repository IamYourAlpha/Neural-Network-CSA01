using namespace std;
#include <bits/stdc++.h>

#define N                   3
#define R                   3
#define n_sample            3
#define eta                 0.05
#define lambda              1.0
#define desired_error       0.01
//#define sigmoid(x)    (2.0/(1.0+exp(-lambda*x))-1.0)
#define sigmoid(x)       ( (exp(lambda*x) - exp(-lambda*x) ) / (exp(lambda*x) + exp(-lambda*x) ) )
#define frand()             ( rand()%10000/10001.0 )
#define randomize()          srand( (unsigned int)time(NULL))


double x[n_sample][N] = {
    { 10, 2, -1},
    { 2, -5, -1},
    { -5, 5, -1},
};

double d[n_sample][R] = {
    { 1, -1, -1},
    { -1, 1, -1},
    { -1, -1, 1},
};


double w[R][N];
double o[R];

void initilization();
void FindOutput();
void PrintResult();


void initialization() {

    randomize();
    for ( int k = 0; k < R; k++) {
        for (int l = 0; l < N; l++) {
            w[k][l] = frand() - 0.5;
        }
    }
}

void FindOutput(int index) {

    int i,j;
    double tot=0.0;

    for(i = 0; i < R; i++) {
        tot = 0.0;
        for(j = 0; j < N; j++) {
            tot += w[i][j] * x[index][j];
        }
        o[i] = sigmoid(tot);
    }
}

void PrintResult() {

    int i,j;
     printf("\n****Neuron Output*******\n");
    for(i = 0; i < n_sample; i++) {
        FindOutput(i);
        puts("");
        for(j = 0; j < R; j++)
            printf("%f          ", o[j]);
    }
    printf("\n\n*****Weights******\n");
    for(int i = 0; i < R; i++){
        for(int j = 0; j < N; j++) {
            printf("%f      ", w[i][j]);
        }
        printf("\n");
    }
}

int main() {

    int i, j, p, q = 0;
    double Error = DBL_MAX;
    double delta;
    initialization();

    while ( Error > desired_error ) {

        q++;
        Error = 0;
        for( p = 0; p < n_sample; p++) {
            FindOutput(p);
            for( i = 0; i < R; i++ )
                Error += 0.5 * pow(d[p][i] - o[i], 2.0);
            for( i = 0; i < R; i++) {
                delta = (d[p][i] - o[i]) * (1- o[i]*o[i])/2;
                for(j = 0; j < N; j++) {
                    w[i][j] += eta * delta * x[p][j];
                }
            }
        }
        printf("Error is the %d-th epoch = %f\n", q, Error);
    }
    PrintResult();
    return 0;
}











