#ifndef EXAMPLE_H
#define EXAMPLE_H

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <stdbool.h>

//smc/Exp

typedef enum ExprType{Constant,Variable,Expr1,StepBoundExpr1,Expr2}ExprType;
typedef enum UnOp{Neg,Not,Future,Globally,Next}UnOp;
typedef enum BinOp{Imp,And,Or,Eq,Neq,Lt,Le,Gt,Ge,Add,Min,Mul,Div}BinOp;    //,Until,WeakUntil

typedef struct action{
    int commandtypeVarindex[2];
    int assignRandom;
    long condrandom;
}action;

typedef struct Expr
{
    ExprType type;
    UnOp uop;
    BinOp bop;
    
    int stepbound;
    char* name;
    int value;
    
    struct Expr* child;
    struct Expr* left;
    struct Expr* right;
}Expr;

//smc/state

typedef struct State{
    int numvar;
    int *varvalue;
    char**varname;
}State;

//treenode.h

extern const int numprivatevars;    //number of public variables of all programs
extern const int numpublicvars;        //number of private variables of each program
extern int progid;
extern const int maxconst;
extern const int numprog;        //number of programs
extern int mutype;

typedef struct exp_
{
    int type;    //0:CONST  1:VAR
    int index;    //0:value of CONST  1:index of VAR
    //0:v[0],1:v[1],2:v[2],3:v[me],4:v[other]
    int number;
}exp_;

typedef struct cond
{
    int type;    //0:TRUE  1:FALSE  2:AND  3:OR  4:NOT            -1:wi
    //5:EQ    6:NEQ    7:GEQ  8:GT
    
    //new -1:wi;    0:True;     1EQ;    2:NEQ     3:AND    4:OR
    
    struct exp_* exp1;
    struct exp_* exp2;
    struct cond* cond1;
    struct cond* cond2;
    int number;
}cond;
typedef struct treenode
{
    int type;//0:IF  1:WHILE  2:SEQ no3 4:ASGN 5:critical section
    int index;//ASSIGN left VAR index
    cond* cond1;
    struct treenode* treenode1;
    struct treenode* treenode2;
    struct treenode* next;
    struct treenode* parent;
    exp_* exp1;
    int goodexamples;
    int badexamples;
    int depth;
    int height;
    int numofstatements;
    int fixed;
    int pc;    //rml
    int number;
}treenode;

typedef struct program
{
    treenode* root;
    int maxdepth;
    int progid;
    int maxconst;
    int numprivatevars;
    int numpublicvars;
    double fitness;
    int numofevent[7];
    double propertyfit[7];
    int checkedBySpin;
    int illegal;
}program;

typedef struct organism
{
    program** progs;
    double fitness;
    //int enterCS;
}organism;

//trace.h

typedef struct trace
{
    int steplength;
    int numofprog;
    int numofvar;
    program** root;                //numofprog * program*
    //int** valueofvar;            //steplength * numofvar * int
    int* executeprogid;            //steplength * int
    treenode** executenode;        //steplength* treenode*
    int satisfied;
    State** states;                //(steplength + 1) * State*
    int tracetype;
}trace;

//smc/Exp

Expr* createExprConstant(int value);
Expr* createExprVar(char* name);
Expr* createExprExp1(UnOp op,Expr* child);
Expr* createExprStepb(UnOp op,Expr* child,int stepbound);
Expr* createExprExpr2(BinOp op,Expr* left,Expr* right);

Expr* generateStepExpr(char* buf, int leftindex, int rightindex);

int getStepBound(Expr* exp);

//smc/state
int getVarindexFromState(State* st,char* name);
int getVarvalueFromState(State* st,char* name);

//treenode.h
bool checkIfFixed(treenode* t);
bool statementsTooLong(treenode* t);
void printType(treenode* t);
int existCS(treenode* t);
exp_* createExp(int t,int ind);
cond* createCond(int t,exp_* e1,exp_* e2,cond* c1,cond* c2);
treenode* createTreenode(int t,int ind,cond* c,treenode* t1,treenode* t2,exp_* e);

int nextrand(int range);
void printprog(treenode* root,int blank,program* prog);
void orgToPml(organism* org,FILE* f);

exp_* genexp(program* prog,int type);
cond* gencond(program* prog,int type);
treenode* genprog(int depth,program* prog);
int* getPublicVarUsed(treenode* prog,int num);
treenode* genCS(int depth,program* prog);
void addCS(program* prog);
int equalExp(exp_* e1,exp_* e2);

void setNumOfStatements(treenode* root);
void setLinesTreenode(treenode* t,int depth);
int setFixed(treenode* t);
int getFixed(treenode* t);
void setAll(program* prog);

exp_* copyExp(exp_* e);
cond* copyCond(cond* c);
treenode* copyTreenode(treenode* t);
program* copyProgram(program* prog);

program* genProgram(program* templat,int progid);
organism* genOrganism(program* template);
program** genInitTemplate(int num);

void freeAll(organism* org,program* prog,treenode* t,cond* c,exp_* e,int type);
int compareTreenode(treenode* t1,treenode* t2);
//treenode* example_d();
//treenode* example_e();
//treenode* wrongcase_1();
//treenode* wrongcase_2();

char* newprintprog(treenode* root,program* prog, char* s);
void orgTorml(organism* org, FILE* f);

//trace.h

void setbadexamples(trace** t,int num);
void setNext(treenode* root);
void setParent(treenode* root);

int checkEnterCS(trace* t);
double calculateFitness(organism* prog,Expr** exp,int numexp,double* coef);
void calculateFitness2(organism* prog,int type);

void initTraceGlobalVar(int steplength);
void freeTrace(trace* t);
//treerank
typedef struct mutationNode
{
    treenode* node;
    cond* cond;
    exp_* exp;
    int ranksum;
    int type;
}mutationNode;

typedef struct treeRank
{
    mutationNode** candidate;
    int numcandidate;
    int maxnumcandidate;
    int ranksum;
}treeRank;

void addNode(treeRank* tr,treenode* node);
void searchNode(treenode* root,treeRank* tr,int type,int maxdepth);


//mutation.h


int satisfyMutationReduction(treenode* t);
void mutationCond(cond* root,program* prog,int type);
//void mutationExp(exp* root,program* prog);
program* mutation(program* parent);

program** genNewCandidate(int numofcandidate,program** candidate,int numofmutation);
program** genNewCandidateWithCoefficient1(int numofcandidate,program** candidate,int numofmutation,double coef);
program** genNewCandidateWithCoefficient2(int numofcandidate,program** candidate,int numofmutation,double coef);

program** selectNewCandidate(int numofcandidate,program** candidate,int numofmutation, program** newcandidate);
program** selectNewCandidateWithFitness(int numofcandidate,program** candidate,int numofmutation, program** newcandidate);

bool existNullCond(treenode* t);

mutationNode* chooseNode(treeRank* tr);
double* set_coef(int numofrequirements);
extern Expr** set_requirments(int numofrequirements);
void setExpNum(exp_* e,program* prog, int number);
int setCondNum(cond* c,program* prog, int number);
int setTreenodeNum(treenode* root, program* prog, int number);
//void newprintprog(treenode* root,program* prog, char *s);
void newprintcond(cond* c,program* prog);
char* newprintexp(exp_* e,program* prog, int node, char* s);
//void newprintexp(exp_* e,program* prog, int node);
char* newprintexp_(exp_* e,program* prog, char* s);
//void newprintexp_(exp_* e,program* prog);
void expNode(exp_* e,program* prog);
char* condNode(cond* c,program* prog, int node, char* s);
char* treeNode_(treenode* root,program* prog, int node, char* s);
int compareNode(treenode* root,int type,int maxdepth);
char* iToStr(int number);
treenode* findNode(treenode* root, program* prog, int number);
program* mutation1(program* parent, int nodeNum, int actionType);
void printAst(program* prog);
//extern program** initProgram(int numofcandidate);

extern double My_variable;
//extern int numprog;
extern double genseq;
extern int fact(int n);
extern int my_mod(int x, int y);
extern char *get_time();
long setConrandom(int a, int b, int c, int d, int e, int f, int g);
action* setAction(int layer, int actionType);
program* mutation_(program* candidate0, int nodeNum, int actType,Expr** requirements ,int numofrequirements,double* coef);
program* initProg(Expr** requirements ,int numofrequirements,double* coef);
void outputLog_(char* str);
int mutationCond_(cond* root,program* prog,int type, int action);
int* getLegalAction2(program* parent, int nodeNum);
int spin_(program* candidate);
#endif
