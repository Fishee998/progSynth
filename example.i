%module example

%{
#include "example.h"
%}

%inline %{
/* array[i] */
program* get_program(program** candidate,int number)
{
    return candidate[number];
}

program* neicun()
{
    program* newcandidate = (program*)malloc(sizeof(program));
    return newcandidate;
}
program* set_fitness(program* org, double fitnessValue)
{
    org->fitness = fitnessValue;
    return org;
}
double get_fitness(program* org)
{
    return org->fitness;
}
%}

extern program* mutation1(program* parent, int nodeNum, int actionType);
extern void printAst(program* prog);
extern double* set_coef(int numofrequirements);
extern Expr** set_requirments(int numofrequirements);
extern double My_variable;
extern int fact(int n);
extern int my_mod(int x, int y);
extern char *get_time();
extern program** genInitTemplate(int num);
extern organism* genOrganism(program* templat);
extern double calculateFitness(organism* prog,Expr** exp,int numexp,double* coef);
extern void freeAll(organism* org,program* prog,treenode* t,cond* c,exp_* e,int type);
extern void setAll(program* prog);
extern program* initProg(Expr** requirements ,int numofrequirements,double* coef);
extern program* mutation_(program* candidate0, int nodeNum, int actType,Expr** requirements ,int numofrequirements,double* coef);
