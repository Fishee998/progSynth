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

int illegal(program* org)
{
    return org->illegal;
}
int action2Len(int* action2)
{
    return sizeof(action2);
}
int get_action2(int* action2, int index)
{
    return action2[index];
}
double getPropertyfit(program* org)
{
    return org->propertyfit[0];
}
int getCheckedBySpin(program* org)
{
    return org->checkedBySpin;
}
%}

extern int spin_(program* candidate);
extern int* getLegalAction2(program* parent, int nodeNum);
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
