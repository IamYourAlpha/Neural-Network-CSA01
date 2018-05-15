using namespace std;
#include<bits/stdc++.h>

#define I             3
#define n_sample      4
#define eta           0.05
#define lambda        1.0
#define desired_error 0.001
//#define sigmoid(x)    (2.0/(1.0+exp(-lambda*x))-1.0)
//#define sigmoid(x)       ( (exp(lambda*x) - exp(-lambda*x) ) / (exp(lambda*x) + exp(-lambda*x) ) )
#define frand()       (rand()%10000/10001.0)
#define randomize()   srand((unsigned int)time(NULL))

double x[n_sample][I]={
  { 0, 0, -1},
  { 0, 1, -1},
  { 1, 0, -1},
  { 1, 1, -1},

};

double w[I];
double d[n_sample]={0, 0, 0, 1};
double o;
double bias = -0.1;

void Initialization(void);
void FindOutput(int);
void PrintResult(void);

main(){
  int    i,p,q=0;
  double delta,Error=DBL_MAX;

  Initialization();
  while(Error>desired_error){
    q++;
    Error=0;
    for(p=0; p<n_sample; p++){
      FindOutput(p);
      Error+=0.5*pow(d[p]-o,2.0);
	bias += delta*eta*1.0;
      for(i=0;i<I;i++){
	delta=(d[p]-o);
	w[i]+=eta*delta*x[p][i];
      }
      printf("Error in the %d-th learning cycle=%f\n",q,Error);
    }
  }
  PrintResult();
}

/*************************************************************/
/* Initialization of the connection weights                  */
/*************************************************************/
void Initialization(void){
  int i;

  randomize();
  for(i=0; i<I; i++) w[i]=frand();
}

/*************************************************************/
/* Find the actual outputs of the network                    */
/*************************************************************/
void FindOutput(int p){
  int    i;
  double temp=0;
  for(i=0;i<I;i++) temp += w[i]*x[p][i];

  temp += bias;
  if ( temp < 0) o = 0.0;
  else o = 1.0;
}

/*************************************************************/
/* Print out the final result                                */
/*************************************************************/
void PrintResult(void){
  int i;

  printf("\n\n");
  printf("The connection weights of the neurons: ");
  for(i=0;i<I;i++) printf(" %5f ",w[i]);
  printf("\nbias = %f", bias);
  printf("\n\n**************Output*************\n");
  for(i=0;i<n_sample;i++) FindOutput(i),printf("%0.0f AND %0.0f == %0.0f\n", x[i][0], x[i][1], o);
  printf("\n\n");
}

