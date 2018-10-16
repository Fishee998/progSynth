//
//  example.c
//  progSynByRlSmc
//
//  Created by 媛庄 on 2018/9/10.
//  Copyright © 2018年 媛庄. All rights reserved.
//

#include "example.h"



const int numprog = 2;
double genseq = 0.4;
int mutype = 0;
double My_variable = 3.0;

//mutation 1

//treerank.c
void addNode(treeRank* tr,treenode* node)
{
    if(node == NULL)
        return;
    
    if(tr->numcandidate == tr->maxnumcandidate)
    {
        tr->maxnumcandidate *= 2;
        mutationNode** new = (mutationNode**)malloc(sizeof(mutationNode*) * tr->maxnumcandidate);
        int i;
        for(i = 0;i < tr->numcandidate;i++)
        {
            new[i] = tr->candidate[i];
        }
        free(tr->candidate);
        tr->candidate = new;
    }
    
    mutationNode* p = (mutationNode*)malloc(sizeof(mutationNode));
    p->node = node;
    
    tr->ranksum += node->badexamples;
    p->ranksum = tr->ranksum;
    tr->candidate[tr->numcandidate] = p;
    tr->numcandidate++;
}

void searchNode(treenode* root,treeRank* tr,int type,int maxdepth)
{
    
    if(root == NULL)
        return;
    if(type == 1)
    {
        if(root->fixed == 0 || root->type == 1)        //while
            addNode(tr,root);
    }
    else if(type == 2)        //??
    {
        if(root->depth != 1 && root->numofstatements < 6 && root->depth == 2)
            addNode(tr,root);
        else
            if(root->depth != 1 && root->numofstatements < 2 && root->depth != 2)
                addNode(tr,root);
    }
    else if(type == 3)
    {
        if(satisfyMutationReduction(root) == 1)
            addNode(tr,root);
    }
    else if(type == 4)
    {
        /*if(root->type == 0 && root->treenode2 != NULL)
         {
         if(root->treenode2->fixed == 0)
         addNode(tr,root->treenode2);
         }
         else */
        if(root->type == 1 || root->type == 2)
        {
            if(root->treenode1 != NULL && root->treenode1->fixed == 0 || root->treenode1 == NULL)
                addNode(tr,root);
        }
    }
    else if(type == 5)
    {
        if(root->depth + 2 <= maxdepth)            //!!
            addNode(tr,root);
    }
    searchNode(root->treenode1,tr,type,maxdepth);
    searchNode(root->treenode2,tr,type,maxdepth);
}


int compareNode(treenode* root,int type,int maxdepth)
{
    
    if(type == 1)
    {
        if(root->fixed == 0 || root->type == 1)        //while
            return 0;
    }
    else if(type == 2)        //??
    {
        if(root->depth != 1 && root->numofstatements < 6 && root->depth < maxdepth)
            return 0;
    }
    else if(type == 3)
    {
        if(satisfyMutationReduction(root) == 1)
            return 0;
    }
    else if(type == 4)
    {
        /*if(root->type == 0 && root->treenode2 != NULL)
         {
         if(root->treenode2->fixed == 0)
         addNode(tr,root->treenode2);
         }
         else */
        if(root->type == 1 || root->type == 2)
        {
            if(root->treenode1 != NULL && root->treenode1->fixed == 0 || root->treenode1 == NULL)
                return 0;
        }
    }
    else if(type == 5)
    {
        if(root->depth + 2 <= maxdepth)            //!!
            return 0;
    }
    return 0;
}


mutationNode* chooseNode(treeRank* tr)
{
    if(tr->ranksum == 0)
        return NULL;
    //printf("start tr->ranksum:%d\n",tr->ranksum);
    int p = nextrand(tr->ranksum);
    int i;
    mutationNode* result;
    for(i = 0;i < tr->numcandidate;i++)
    {//printf("i:%d\n",i);
        result = tr->candidate[i];
        if(result == NULL)printf("result NULL!!\n");
        if(result->ranksum > p)
            break;
    }
    //printf("end\n");
    for(i = 0;i < tr->numcandidate;i++)
        if(tr->candidate[i] != result)
            free(tr->candidate[i]);
    free(tr);
    
    return result;
}



//trace.c
int *varvalue;
treenode** nextstep;
treenode** currentstep;
trace* gtrace = NULL;

const int numofcheck = 300;
const int numpublicvars = 1;
const int numprivatevars = 1;
const int maxconst = 2;

int mutype;
int condnull = 0;
void freeTrace(trace* t)
{
    //int i;
    //for(i = 0;i < t->steplength;i++)
    //    free(t->valueofvar[i]);
    //free(t->valueofvar);
    free(t->executeprogid);
    free(t->executenode);
    free(t);
}

void setbadexamples(trace** traces,int num)
{
    
    int i,j;
    for(i = 0;i < num;i++)
    {
        for(j = 0;j < traces[i]->steplength;j++)
        {
            if(traces[i]->satisfied)
                traces[i]->executenode[j]->goodexamples++;
            else
                traces[i]->executenode[j]->badexamples++;
        }
        
    }
}

void setNext(treenode* root)
{
    if(root == NULL)
        return;
    
    if(root->type == 0)
    {
        if(root->treenode1 != NULL)
            root->treenode1->next = root->next;
        if(root->treenode2 != NULL)
            root->treenode2->next = root->next;
    }
    else if(root->type == 1 || root->type == 2)
    {
        if(root->treenode1 != NULL)
            root->treenode1->next = root;
    }
    else if(root->type == 3)
    {
        root->treenode1->next = root->treenode2;
        root->treenode2->next = root->next;
    }
    setNext(root->treenode1);
    setNext(root->treenode2);
}

void setParent(treenode* root)
{
    if(root != NULL)
    {
        if(root->treenode1 != NULL)
        {
            root->treenode1->parent = root;
            setParent(root->treenode1);
        }
        if(root->treenode2 != NULL)
        {
            root->treenode2->parent = root;
            setParent(root->treenode2);
        }
    }
}

int getvarvalue(exp_* e)
{
    if(e == NULL)
    {
        printf("getvarvalue null!\n");
        return -1;
    }
    
    if(e->type == 0)
        return e->index;
    if(e->type == 1)
        return varvalue[e->index];
    return 0;
    
}

bool getcondvalue(cond* c)
{
    if(c == NULL)
    {
        printf("getcondvalue null!\n");
        condnull = 1;
        return false;
    }
    if(c->type == -1)
        printf("condition type = -1\n");
    switch(c->type)
    {
        case -1:if(nextrand(10) < 8) return true;else return false;
        case 0:return true;
        case 1:return (getvarvalue(c->exp1) == getvarvalue(c->exp2));
        case 2:return (getvarvalue(c->exp1) != getvarvalue(c->exp2));
        case 3:return (getcondvalue(c->cond1) && getcondvalue(c->cond2));
        case 4:return (getcondvalue(c->cond1) || getcondvalue(c->cond2));
        default:printf("getcondvalue type error\n");return false;
    }
}

void printvarvalue()
{
    
    int numofvars = numpublicvars + numprivatevars * numprog;
    int i;
    for(i = 0;i < numpublicvars;i++)
    {
        printf("v%d=%d",i,varvalue[i]);
        if(i < 10)
            printf("  ");
        else
            printf(" ");
        
        if(i % 8 == 7)
            printf("\n");
    }
    printf("\n");
    for(i = 0;i < numprog;i++)
    {
        int j;
        for(j = 0;j < numprivatevars;j++)
        {
            printf("p%d.v%d=%d",i,j,varvalue[numpublicvars + i * numprivatevars + j]);
            
            if(i < 10 && j < 10)
                printf("   ");
            else if(i < 10 || j < 10)
                printf("  ");
            else
                printf(" ");
            
            if(j % 8 == 7)
                printf("\n");
        }
        printf("\n");
    }
    printf("\n");
}

void printTrace(trace* t,int length)
{
    int i;
    printf("\nprinttrace,length:%d\n",t->steplength);
    for(i = 0;i < length && i < t->steplength;i++)
    {
        printf("executeprogid:%d, type:%d, varvalue:",t->executeprogid[i],t->executenode[i]->type);
        if(t->executenode[i]->next != NULL)
            printf("next type:%d",t->executenode[i]->next->type);
        printf("\n");
        //int j;
        //for(j = 0;j < t->numofvar;j++)
        //    printf("v[%d]=%d;",j,t->valueofvar[i][j]);
        printf("\n");
    }
    
}

bool checkVarValueChanged(program* prog)
{
    free(varvalue);
    varvalue = (int*)malloc(numpublicvars + numprivatevars * numprog);
    treenode* next = prog->root;
    treenode* current = NULL;
    
    int i;
    for(i = 0;i < numpublicvars + numprivatevars * numprog;i++)
        varvalue[i] = 0;
    int count = 0,step = 0;
    while(count < 2 && step < 50)
    {
        step++;
        while(next->type == 3)
            next = next->treenode1;
        current = next;
        bool condition;
        //printf("type=%d\n",next->type);
        switch(next->type)
        {
            case 0:    condition = getcondvalue(next->cond1);
                if(condition)
                    next = next->treenode1;
                else
                    next = next->next;
                break;
            case 1:    if(next->cond1->type == -1)
            {
                count++;
                condition = true;
            }
            else
                condition = getcondvalue(next->cond1);
                
                if(condition)
                {
                    if(next->treenode1 != NULL)
                        next = next->treenode1;
                }
                else
                    next = next->next;
                break;
            case 4:    varvalue[next->index] = getvarvalue(next->exp1);
                next = next->next;
                break;
            case 5: next = next->next;
                break;
        }
    }
    
    if(step < 50 && varvalue[0] == 0 && varvalue[1] == 0)
        return false;
    else
        return true;
}

trace* gettrace(organism* org,int num)
{
    //printf("init state:\n");
    //printvarvalue();
    
    /*  trace* gtrace = (trace*)malloc(sizeof(trace));
     gtrace->steplength = steplength;
     gtrace->numofprog = numprog;
     gtrace->numofvar = numpublicvars + numprivatevars * numprog;
     */
    gtrace->root = org->progs;
    //gtrace->valueofvar = (int**)malloc(sizeof(int*) * steplength);
    /*    gtrace->executeprogid = (int*)malloc(sizeof(int) * steplength);
     gtrace->executenode = (treenode**)malloc(sizeof(treenode*) * steplength);
     gtrace->satisfied = nextrand(2) == 0 ? true : false;
     
     free(varvalue);
     free(nextstep);
     free(currentstep);
     varvalue = (int*)malloc(sizeof(int) * gtrace->numofvar);
     nextstep = (treenode**)malloc(sizeof(treenode*) * gtrace->numofprog);
     currentstep = (treenode**)malloc(sizeof(treenode*) * gtrace->numofprog);
     */
    
    int i,j;
    int step = gtrace->steplength;            //???
    //int* stepincs = (int*)malloc(sizeof(int) * numprog);
    
    for(i = 0;i < numprog;i++)
    {
        nextstep[i] = gtrace->root[i]->root;
        currentstep[i] = NULL;
    }
    for(i = 0;i < gtrace->numofvar;i++)
        varvalue[i] = 0;
    
    
    //printf("22\n");
    for(i = 0;i < step;i++)
    {
        bool allprogdone = true;
        for(j = 0;j < numprog;j++)
        {
            if(nextstep[j] != NULL)
            {
                allprogdone = false;
            }
        }
        if(allprogdone)
        {
            gtrace->steplength = i;        //!!
            break;
        }
        
        //printf("23\n");
        int executeprogid;
        
        /*    if(nextstep[0] == NULL)
         executeprogid = 1;
         else if(nextstep[1] == NULL)
         executeprogid = 0;
         else if(currentstep[0] != NULL && currentstep[0]->type == 5)
         {    //printf("231\n");
         stepincs[0]++;
         if(stepincs[0] > 20)
         {
         stepincs[0] = 0;
         executeprogid = nextrand(2);
         }
         else
         executeprogid = 1;
         //printf("232\n");
         }
         else if(currentstep[1] != NULL && currentstep[1]->type == 5)
         {
         //printf("233\n");
         stepincs[1]++;
         if(stepincs[1] > 20)
         {
         stepincs[1] = 0;
         executeprogid = nextrand(2);
         }
         else
         executeprogid = 0;
         //printf("234\n");
         }
         else
         executeprogid = nextrand(2);*/
        
        if(nextstep[0] == NULL)
            executeprogid = 1;
        else if(nextstep[1] == NULL)
            executeprogid = 0;
        else if(currentstep[0] != NULL && currentstep[0]->type == 5)
        {
            if(nextrand(10) < 9)
                executeprogid = 1;
            else
                executeprogid = 0;
        }
        else if(currentstep[1] != NULL && currentstep[1]->type == 5)
        {
            if(nextrand(10) < 9)
                executeprogid = 0;
            else
                executeprogid = 1;
        }
        else
            executeprogid = nextrand(2);
        
        while(nextstep[executeprogid]->type == 3)
            nextstep[executeprogid] = nextstep[executeprogid]->treenode1;
        
        currentstep[executeprogid] = nextstep[executeprogid];
        gtrace->executeprogid[i] = executeprogid;
        gtrace->executenode[i] = currentstep[executeprogid];
        //printf("24\n");
        bool condition;
        switch(nextstep[executeprogid]->type)
        {
            case 0:    condition = getcondvalue(nextstep[executeprogid]->cond1);
                if(condition)
                    nextstep[executeprogid] = nextstep[executeprogid]->treenode1;
                else
                    nextstep[executeprogid] = nextstep[executeprogid]->next;
                break;
            case 1:    if(nextstep[executeprogid]->cond1->type == -1)
            {
                if(executeprogid == 1 && num < numofcheck * 0.85 || executeprogid == 0 && (num < numofcheck * 0.7 || num >= numofcheck * 0.85))//num of check
                    condition = true;
                else
                    condition = false;
                
                //if(condition)
                //    printf("prog:%d,num:%d,true\n",executeprogid,num);
                //else
                //    printf("prog:%d,num:%d,false\n",executeprogid,num);
            }
            else
                condition = getcondvalue(nextstep[executeprogid]->cond1);
                
                if(condition)
                {
                    if(nextstep[executeprogid]->treenode1 != NULL)
                        nextstep[executeprogid] = nextstep[executeprogid]->treenode1;
                }
                else
                    nextstep[executeprogid] = nextstep[executeprogid]->next;
                break;
                /*case 2:    if(nextstep[executeprogid]->index == 0)
                 {
                 nextstep[executeprogid]->index = 1;
                 nextstep[executeprogid] = nextstep[executeprogid]->treenode1;
                 break;
                 }
                 condition = getcondvalue(nextstep[executeprogid]->cond1);
                 if(condition)
                 nextstep[executeprogid] = nextstep[executeprogid]->treenode1;
                 else
                 {
                 nextstep[executeprogid]->index = 0;
                 nextstep[executeprogid] = nextstep[executeprogid]->next;
                 }
                 break;*/
                //case 3:    nextstep[executeprogid] = nextstep[executeprogid]->treenode1;
                //        break;
            case 4://printprog(nextstep[executeprogid],0);
                varvalue[nextstep[executeprogid]->index] = getvarvalue(nextstep[executeprogid]->exp1);
                nextstep[executeprogid] = nextstep[executeprogid]->next;
                break;
            case 5: nextstep[executeprogid] = nextstep[executeprogid]->next;
                //printf("critical section!\n");
                break;
        }
        //printf("step:%d,exeprogid:%d\n",i,executeprogid);
        
        //printvarvalue();
        /*gtrace->valueofvar[i] = (int*)malloc(sizeof(int) * gtrace->numofvar);
         for(k = 0;k < gtrace->numofvar;k++)
         {
         gtrace->valueofvar[i][k] = varvalue[k];
         }*/
    }
    
    return gtrace;
}

int checkEnterCS(trace* t)        //M:1 B1:2 B2:3 O1:4 O2:5
{    //printf("a%d\n",t->steplength);
    int a = -1,b = -1;            //0:want to enter    1:enter
    //int a = 0,b = 0;
    int* inCS = (int*)malloc(sizeof(int) * t->numofprog);    // * 2
    int i;
    for(i = 0;i < t->numofprog;i++)
        inCS[i] = 0;
    for(i = 0;i < t->steplength;i++)
    {
        //printf("b i=%d\n",i);
        //printf("%d\n",t->executeprogid[i]);
        //printf("bb%d\n",t->executenode[i]->type);
        //if(t->executenode[i] == NULL)    printf("error!!!\n");
        //printf("bbb\n");
        int p = -1;
        if(t->executenode[i]->type == 5)
        {
            inCS[t->executeprogid[i]] = 1;
            p = t->executeprogid[i];
            //printf("p = %d\n",p);
        }
        else
            inCS[t->executeprogid[i]] = 0;
        
        if(!(t->executenode[i]->type == 1 && t->executenode[i]->cond1->type == -1))
        {
            if(t->executeprogid[i] == 0 && a == -1)
                a = 0;
            else if(t->executeprogid[i] == 1 && b == -1)
                b = 0;
        }
        
        
        if(inCS[0] == 1 && inCS[1] == 1)
        {
            free(inCS);
            return 0;
        }
        else if(inCS[0] == 1 && inCS[1] == 0 && p == 0)
            a++;
        else if(inCS[0] == 0 && inCS[1] == 1 && p == 1)
            b++;
        //if(p > -1)
        //    printf("p=%d,a=%d,b=%d\n",p,a,b);
    }
    
    free(inCS);
    //printf("a=%d,b=%d",a,b);
    if(a >= 2 && b >= 2)
        return 2;
    else if(a >= 2 && b >= 0 && b <= 1|| a >= 0 && a <= 1 && b >= 2)
        return 3;
    else if(a * b < -1)
        return 4;
    else if(a * b == -1)
        return 5;
    else if(a >= 0 && a <= 1 && b >= 0 && b <= 1)
        return 6;            //B3
    else if(a == 0 && b == -1 || a == -1 && b == 0)
        return 7;            //O3
    else
        return 1;
}


int* getLegalAction2(program* parent, int nodeNum)
{
    program* newprog = copyProgram(parent);
    newprog->checkedBySpin = 0;
    treenode* chnode = findNode(newprog->root, newprog, nodeNum);
    if (chnode == NULL) {
        newprog->illegal = 1;
        printf("treenode1 error");
    }
    //Replacement Mutation type
    treenode* mnode = chnode;
    int* action = (int*)malloc(sizeof(int)*50);
    int index = 0;
    for (int actionNum = 0; actionNum < 50; actionNum++) {
        if (actionNum >= 0 && actionNum < 3 )
        {
            if (mnode->fixed == 0) {
                switch (actionNum) {
                    case 0:
                        action[index] = actionNum;
                        index++;
                        break;
                    case 1:
                        if (mnode->depth == 3) {
                            action[index] = actionNum;
                            index++;
                        }
                        break;
                    case 2:
                        if (mnode->depth == 3) {
                            action[index] = actionNum;
                            index++;
                        }
                        break;
                }
            }
        }
        else if(actionNum >= 3 && actionNum < 31)
        {
            if(mnode->cond1 != NULL)
            {
                if (actionNum >= 3 && actionNum < 12)
                {
                    if(mnode->cond1->type == 1 || mnode->cond1->type == 2 || mnode->cond1->type == 3 || mnode->cond1->type == 4)
                    {
                    action[index] = actionNum;
                    index++;
                    }
                }
                else if (actionNum >= 12 && actionNum < 21)
                {
                    if (mnode->cond1->type == 3 || mnode->cond1->type == 4)
                    {
                        action[index] = actionNum;
                        index++;
                    }
                }
                else if(actionNum >= 21 && actionNum < 25)
                {
                    if (mnode->cond1->type == 1 || mnode->cond1->type == 2)
                    {
                        action[index] = actionNum;
                        index++;
                    }
                }
                else if (action >= 25 && action < 31)
                {
                    if (mnode->cond1->type == 3 || mnode->cond1->type == 4)
                    {
                        action[index] = actionNum;
                        index++;
                    }
                }
            }
        }
        else if(actionNum >= 31 && actionNum < 40)
        {
            if (mnode->exp1 != NULL)
            {
                action[index] = actionNum;
                index++;
            }
        }
        else if(actionNum >= 40 && actionNum < 42)
        {
            if(mnode->fixed == 1 || mnode->depth + mnode->height == newprog->maxdepth + 1 || mnode->depth == 2)
            {
                int illegal = 1;
            }
            else
            {
                action[index] = actionNum;
                index++;
            }
        }
        else if(actionNum >= 42 && actionNum < 48 )
        {
            if(!(mnode->depth != 1 && mnode->numofstatements < 6 && mnode->depth == 2) && !(mnode->depth != 1 && mnode->numofstatements < 2 && mnode->depth != 2))
            {
                int illegal = 1;
            }
            else
            {
                action[index] = actionNum;
                index++;
            }
        }
        else if(actionNum >= 48 && actionNum < 50)
        {
            switch (actionNum) {
                case 48:
                    if (mnode->treenode1 != NULL && mnode->treenode1->fixed == 0 && (mnode->treenode2 == NULL || mnode->treenode2->fixed == 0))
                    {
                        action[index] = actionNum;
                        index++;
                    }
                    break;
                case 49:
                    if (mnode->treenode2 != NULL && mnode->treenode2->fixed == 0 && (mnode->treenode1 == NULL && mnode->treenode1->fixed == 0 ))
                    {
                        action[index] = actionNum;
                        index++;
                    }
                    break;
            }
        }
        action[index] = '\n';
        
    }
    return action;
}

void calculateFitness2(organism* prog,int type)
{
    int M = 0,B1 = 0,B2 = 0,O1 = 0,O2 = 0,B3 = 0,O3 = 0;//,BM = 0;
    double a = 0.6,b = 0.2,c = 0,d = 0.2,e = -0.1;//,f = 0.7;
    int i;
    for(i = 0;i < numofcheck;i++)
    {
        if(existNullCond(prog->progs[0]->root))
            printf("exist null condition\n");
        trace* t = gtrace;
        if(condnull == 1)
        {
            printf("ERROR!\n%d,program0\n",i);
            printprog(prog->progs[0]->root,0,prog->progs[0]);
            printf("%d,program1\n",i);
            printprog(prog->progs[1]->root,0,prog->progs[1]);
            if(existNullCond(prog->progs[0]->root))
                printf("\nexist null condition");
            printf("\n");
            exit(-1);
        }
        
        int enter = checkEnterCS(t);
        
        //freeTrace(t);    //!!
        switch(enter)
        {
            case 1:M++;break;
            case 2:M++;B1++;break;
            case 3:M++;B2++;break;
            case 4:M++;O1++;break;
            case 5:M++;O2++;break;
            case 6:M++;B3++;break;
            case 7:M++;O3++;break;
        }
    }
    
    int allB = B1 + B2 + B3;
    if(B1 > (double)allB * 0.7)
    {
        B1 = allB;
        B2 = 0;
        B3 = 0;
    }
    prog->fitness = a * (double)M + b * (double)B1 + c * (double)B2 + d * (double)O1 + e * (double)O2 - 0.2 * (double)O3;// + f * (double)BM;
    prog->fitness = prog->fitness / (numofcheck / 100);
    //if(!checkVarValueChanged(prog->progs[0]) && !checkVarValueChanged(prog->progs[1]))
    //    prog->fitness += 3;
    prog->progs[0]->numofevent[0] = M;
    prog->progs[0]->numofevent[1] = B1;
    prog->progs[0]->numofevent[2] = B2;
    prog->progs[0]->numofevent[3] = O1;
    prog->progs[0]->numofevent[4] = O2;
    prog->progs[0]->numofevent[5] = B3;
    prog->progs[0]->numofevent[6] = O3;
    
    if(type == 1)
    {
        printf("M:%d,B1:%d,B2:%d,B3:%d,O1:%d,O2:%d,O3:%d,fitness:%lf\n",M,B1,B2,B3,O1,O2,O3,prog->fitness);
        printf("fitness over\n");
    }
}

void setTraceStates(trace* t,int type)
{
    int tracetypeindex = getVarindexFromState(t->states[0],"tracetype");
    int csindex[2],enterindex[2];
    csindex[0] = getVarindexFromState(t->states[0],"cs0");
    csindex[1] = getVarindexFromState(t->states[0],"cs1");
    enterindex[0] = getVarindexFromState(t->states[0],"enter0");
    enterindex[1] = getVarindexFromState(t->states[0],"enter1");
    //printf("%d,%d,%d,%d,%d\n",tracetypeindex,csindex[0],csindex[1],enterindex[0],enterindex[1]);
    int i,j;
    int lastnodecs[2];
    lastnodecs[0] = 0;
    lastnodecs[1] = 0;
    
    for(j = 0;j < t->states[0]->numvar;j++)
        t->states[0]->varvalue[j] = 0;
    t->states[0]->varvalue[tracetypeindex] = type;
    
    for(i = 0;i < t->steplength;i++)
    {    //printf("%d,",t->executenode[i]->type);
        for(j = 0;j < t->states[0]->numvar;j++)
            t->states[i + 1]->varvalue[j] = t->states[i]->varvalue[j];
        if(i > 0)
        {
            int id = t->executeprogid[i];
            if(t->executenode[i]->type == 5)
            {    //printf("%d,",t->executenode[i]->type);
                t->states[i + 1]->varvalue[csindex[id]] = 1;
                t->states[i + 1]->varvalue[enterindex[id]]++;
                lastnodecs[id] = 1;
            }
            else if(lastnodecs[id] == 1)
            {
                lastnodecs[id] = 0;
                t->states[i + 1]->varvalue[csindex[id]] = 0;
            }
        }
    }
}


double calculateFitness(organism* prog,Expr** exp,int numexp,double* coef)
{
    int i;
    
    
    int M = 0,B1 = 0,B2 = 0,O1 = 0,O2 = 0,B3 = 0,O3 = 0;//,BM = 0;
    double a = 0.6,b = 0.2,c = 0,d = 0.2,e = -0.1;//,f = 0.7;
    
    
    double* result = (double*)malloc(sizeof(double) * numexp);
    for(i = 0;i < numexp;i++)
        result[i] = 0;
    
    /*printf("start check\n");
     for(i = 0;i < numexp;i++)
     {
     printExpr(exp[i]);
     printf("\n");
     }*/
    
    int count = 0;
    for(i = 0;i < numofcheck;i++)
    {
        if(existNullCond(prog->progs[0]->root))
            printf("exist null condition\n");
        trace* t = gettrace(prog,i);
        
        
        
        /*int enter = checkEnterCS(t);
         switch(enter)
         {
         case 1:M++;break;
         case 2:M++;B1++;break;
         case 3:M++;B2++;break;
         case 4:M++;O1++;break;
         case 5:M++;O2++;break;
         case 6:M++;B3++;break;
         case 7:M++;O3++;break;
         }*/
        
        
        int safety = 1;
        if(i < numofcheck * 0.7)
        {
            if(checkEnterCS(t) == 0)
                safety = 0;
        }
        else
            safety = 0;
        
        if(i < numofcheck * 0.7)
            setTraceStates(t,0);
        else if(i >= numofcheck * 0.7 && i < numofcheck * 0.85)
            setTraceStates(t,1);
        else
            setTraceStates(t,2);
        
        
        
        int j;
        for(j = 0;j < numexp;j++)
        {
            double value = getExprValue(exp[j],t,0);
            //if(value > 1)
            //    printf("value:%lf\n",value);
            result[j] += value;
        }
        /*fprintf(fp,"This is trace %d\n",i);
         for(j = 0;j < t->steplength;j++)
         fprintf(fp,"step%d:%d,%d,%d,%d,%d,%d\n",j,t->states[j]->varvalue[0],t->states[j]->varvalue[1],t->states[j]->varvalue[2],t->states[j]->varvalue[3],t->states[j]->varvalue[4],t->executeprogid[j ]);
         //fprintf(fp,"final result:%d\n",r);
         
         
         
         if(safety == 1)
         printf("safety correct!%d\n",count++);    */
    }
    //fclose(fp);
    
    for(i = 0;i < numexp;i++)
    {
        //if(result[i] > 300)
        //    printf("before result:%lf,numofcheck:%d\n",result[i],numofcheck);
        
        result[i] = result[i] / (double)numofcheck;
        //if(result[i] > 1)
        //    printf("after result:%lf,numofcheck:%d\n",result[i],numofcheck);
    }
    double allB = result[1] + result[2] + result[3];
    if(result[1] > allB * 0.7)
    {
        result[1] = allB;
        result[2] = 0;
        result[3] = 0;
    }
    double fitness = 0;
    for(i = 0;i < numexp;i++)
    {
        fitness += coef[i] * result[i];
        prog->progs[0]->propertyfit[i] = result[i];
    }
    
    
    /*int allB0 = B1 + B2 + B3;
     if(B1 > (double)allB0 * 0.7)
     {
     B1 = allB0;
     B2 = 0;
     B3 = 0;
     }
     prog->fitness = a * (double)M + b * (double)B1 + c * (double)B2 + d * (double)O1 + e * (double)O2 - 0.2 * (double)O3;// + f * (double)BM;
     prog->fitness = prog->fitness / (numofcheck / 100);
     //if(!checkVarValueChanged(prog->progs[0]) && !checkVarValueChanged(prog->progs[1]))
     //    prog->fitness += 3;
     prog->progs[0]->numofevent[0] = M;
     prog->progs[0]->numofevent[1] = B1;
     prog->progs[0]->numofevent[2] = B2;
     prog->progs[0]->numofevent[3] = O1;
     prog->progs[0]->numofevent[4] = O2;
     prog->progs[0]->numofevent[5] = B3;
     prog->progs[0]->numofevent[6] = O3;*/
    
    
    return (fitness * 100);
}


void initTraceGlobalVar(int steplength)
{
    gtrace = (trace*)malloc(sizeof(trace));
    gtrace->steplength = steplength;
    gtrace->numofprog = numprog;
    gtrace->numofvar = numpublicvars + numprivatevars * numprog;
    
    varvalue = (int*)malloc(sizeof(int) * gtrace->numofvar);
    nextstep = (treenode**)malloc(sizeof(treenode*) * gtrace->numofprog);
    currentstep = (treenode**)malloc(sizeof(treenode*) * gtrace->numofprog);
    
    //gtrace->valueofvar = (int**)malloc(sizeof(int*) * steplength);
    gtrace->executeprogid = (int*)malloc(sizeof(int) * steplength);
    if(gtrace->executeprogid == NULL)
    {
        printf("failed allocaating!\n");
        perror("do we have an error\n");
    }
    gtrace->executenode = (treenode**)malloc(sizeof(treenode*) * steplength);
    gtrace->satisfied = nextrand(2) == 0 ? true : false;
    
    gtrace->states = (State**)malloc(sizeof(State*) * (steplength + 1));
    int i;
    for(i = 0;i < steplength + 1;i++)
    {
        gtrace->states[i] = (State*)malloc(sizeof(State));
        gtrace->states[i]->numvar = 5;
        gtrace->states[i]->varvalue = (int*)malloc(sizeof(int) * 5);
        gtrace->states[i]->varname = (char**)malloc(sizeof(char*) * 5);
        gtrace->states[i]->varname[0] = "tracetype";
        gtrace->states[i]->varname[1] = "cs0";
        gtrace->states[i]->varname[2] = "cs1";
        gtrace->states[i]->varname[3] = "enter0";
        gtrace->states[i]->varname[4] = "enter1";
    }
    
}



//init.c
int equalExp(exp_* e1,exp_* e2)
{
    if(e1 == NULL || e2 == NULL)
        return 0;
    if(e1->type == e1->type && e1->index == e2->index)
        return 1;
    return 0;
}

exp_* createExp(int t,int ind)
{
    exp_* p = (exp_*)malloc(sizeof(exp_));
    p->type = t;
    p->index = ind;
    return p;
}
cond* createCond(int t,exp_* e1,exp_* e2,cond* c1,cond* c2)
{
    cond* p = (cond*)malloc(sizeof(cond));
    p->type = t;
    p->exp1 = e1;
    p->exp2 = e2;
    p->cond1 = c1;
    p->cond2 = c2;
    return p;
}

treenode* createTreenode(int t,int ind,cond* c,treenode* t1,treenode* t2,exp_* e)
{//printf("createtreenode type:%d",t);
    treenode* p = (treenode*)malloc(sizeof(treenode));
    p->type = t;
    p->index = ind;
    p->cond1 = c;
    p->treenode1 = t1;
    p->treenode2 = t2;
    p->exp1 = e;
    p->goodexamples = 1;
    p->badexamples = 1;
    p->next = NULL;
    p->parent = NULL;
    if(t == 5)
        p->fixed = 1;
    else
        p->fixed = 0;
    return p;
}

int nextrand(int range)    //generate a random number between 0..range-1
{
    int number = rand() % range;
    // printf("range=%d;number=%d;\n",range,number);
    return number;
}

exp_* genexp(program* prog,int vartype)
{
    
    int exptype = nextrand(2);
    if(vartype != -1)
        exptype = vartype;
    int varindex = 0;
    exp_* result = NULL;
    switch(exptype)
    {
        case 0:    result = createExp(exptype,nextrand(prog->maxconst + 2));
            if(result->index >= 2)            //!!
                result->index++;
            break;
        case 1:    varindex = nextrand(prog->numpublicvars + 2 * prog->numprivatevars + 2);//2:numofprog    2:me,other
            result = createExp(exptype,varindex);break;
    }
    return result;
}

exp_* genexp_(program* prog,int vartype, int assignRandom)
{
    //exptype 0 : varindex 0 1 2 3
    //exptype 1 : varindex 0 1 2 3 4
    int exptype = assignRandom / 10;
    int varindex = assignRandom % 10;
    if(vartype != -1)
        exptype = vartype;
    exp_* result = NULL;
    switch(exptype)
    {
        case 0:
            result = createExp(exptype, varindex);
            if(result->index >= 2)            //!!
                result->index++;
            break;
        case 1: //2:numofprog    2:me,other
            result = createExp(exptype,varindex);break;
    }
    return result;
}

cond* gencond(program* prog,int type)    //type==3:true,==,!=    type==5:true,==,!=,&&,||
{
    int condtype = nextrand(type - 1) + 1;
    cond* result = NULL;
    switch(condtype)
    {
        case 0: result = createCond(0,NULL,NULL,NULL,NULL);break;
        case 1: result = createCond(1,genexp(prog,1),genexp(prog,0),NULL,NULL);
            break;
        case 2: result = createCond(2,genexp(prog,1),genexp(prog,0),NULL,NULL);
            break;
        case 3: result = createCond(3,NULL,NULL,gencond(prog,3),gencond(prog,3));
            break;
        case 4: result = createCond(4,NULL,NULL,gencond(prog,3),gencond(prog,3));
            break;
    }
    return result;
}


cond* gencond_(program* prog,int type, long condRandom_)    //type==3:true,==,!=    type==5:true,==,!=,&&,||
{
    long condRandom1 = 0;
    long condRandom2 = 0;
    long condRandom_x = 0;
    int assignRandom1, assignRandom2;
    int condtype = 0;
    if (condRandom_ / 100000000 != 0) {
        condtype = condRandom_ / 100000000;
        if (condtype < 0 || condtype > 4) {
            outputLog_("depth3 condtype34 error");
        }
        condRandom_x = condRandom_ % 100000000;
        condRandom1 = condRandom_x / 10000;
        condRandom2 = condRandom_x % 10000;
    }
    else
    {
        condtype = condRandom_ / 1000;
        if (condtype != 1 && condtype != 2) {
            outputLog_("depth3 condtype12 error");
        }
        condRandom_x = condRandom_ % 1000;
        assignRandom1 = condRandom_x / 10;
        if (assignRandom1 < 10 || assignRandom1 > 14) {
            outputLog_("depth3 assignRandom1 error");
        }
        assignRandom2 = condRandom_x % 10;
        if (assignRandom1 < 0 || assignRandom1 > 3) {
            outputLog_("depth3 assignRandom2 error");
        }
    }
    cond* result = NULL;
    switch(condtype)
    {
        case 0: result = createCond(0,NULL,NULL,NULL,NULL);break;
        case 1: result = createCond(1,genexp_(prog,1,assignRandom1),genexp_(prog,0,assignRandom2),NULL,NULL);
            break;
        case 2: result = createCond(2,genexp_(prog,1,assignRandom1),genexp_(prog,0,assignRandom2),NULL,NULL);
            break;
        case 3: result = createCond(3,NULL,NULL,gencond_(prog,3,condRandom1),gencond_(prog,3,condRandom2));
            break;
        case 4: result = createCond(4,NULL,NULL,gencond_(prog,3,condRandom1),gencond_(prog,3,condRandom2));
            break;
    }
    return result;
}

void outputLog_(char* str)
{
    FILE *fp;
    fp = fopen("./log.txt", "a");
    fprintf(fp, "%s", str);
    fclose(fp);
}

//0:IF  1:WHILE  2:UNTIL  3:SEQ  4:ASGN    5:critical section
//treenode* createTreenode(int t,int ind,cond* c,treenode* t1,treenode* t2,treenode* f,exp* e);
treenode* genprog(int depth,program* prog)
{
    int commandtype;
    int height = prog->maxdepth + 1 - depth;
    
    if(height < 2)
        commandtype = 0;
    else
    {
        //int p = nextrand(1000);
        //if(p < genseq * 1000)
        //    commandtype = 4;
        //else
        // commandtype = nextrand(5);
        commandtype = nextrand(3);
        //if(commandtype == 3)
        //    commandtype = 4;
    }
    
    int varindex = 0;
    
    treenode* result = NULL;
    switch(commandtype)
    {
        case 0: varindex = nextrand(prog->numpublicvars + 1) + prog->numprivatevars * 2;    //??
            result = createTreenode(4,varindex,NULL,NULL,NULL,genexp(prog,0));
            break;
        case 1: result = createTreenode(0,0,gencond(prog,5),NULL,NULL,NULL);    //0:TRUE
            while(result->cond1->type == 0)
            {
                free(result->cond1);
                result->cond1 = gencond(prog,5);
            }
            result->treenode1 = genprog(depth + 1,prog);break;
        case 2: result = createTreenode(1,0,gencond(prog,5),NULL,NULL,NULL);
            while(result->cond1->type == 0)
            {
                free(result->cond1);
                result->cond1 = gencond(prog,5);
            }
            result->treenode1 = genprog(depth + 1,prog);break;
    }
    
    return result;
}
//0:IF  1:WHILE  2:UNTIL  3:SEQ  4:ASGN    5:critical section
treenode* genprog_(int depth,program* prog, int commandtypeVarindex[], int assignRandom, long condRandom, int i)
{
    // commandtypeVarindex: 02 03 10 20
    // commandtype 0 -> varindex 2 3
    int commandtype = commandtypeVarindex[i] / 10;
    int varindex = 0;
    //int assignRandom_ = assignRandom[i];
    //long condRandom_1;
    //long condRandom_2;
    /*
    if (condRandom != NULL) {
        condRandom_1 = condRandom[0];
        condRandom_2 = condRandom[1];
    }
    */
    treenode* result = NULL;
    switch(commandtype)
    {
        case 0:
            varindex = commandtypeVarindex[i] % 10;    //??
            result = createTreenode(4,varindex,NULL,NULL,NULL,genexp_(prog,0,assignRandom));
            break;
        case 1:
            result = createTreenode(0,0,gencond_(prog,5, condRandom),NULL,NULL,NULL);    //0:TRUE
            while(result->cond1->type == 0)
            {
                free(result->cond1);
                result->cond1 = gencond_(prog,5, condRandom);
            }
            result->treenode1 = genprog_(depth + 1, prog, commandtypeVarindex, assignRandom, NULL, 1);break;
        case 2:
            result = createTreenode(0,0,gencond_(prog,5, condRandom),NULL,NULL,NULL);    //0:TRUE
            while(result->cond1->type == 0)
            {
                free(result->cond1);
                result->cond1 = gencond_(prog,5, condRandom);
            }
            result->treenode1 = genprog_(depth + 1,prog, commandtypeVarindex,  assignRandom, NULL, 1);break;
    }
    
    return result;
}

treenode* genCS(int depth,program* prog)
{
    cond* c = createCond(-1,NULL,NULL,NULL,NULL);
    treenode* t1 = createTreenode(3,0,NULL,NULL,NULL,NULL);
    treenode* result = createTreenode(1,0,c,t1,NULL,NULL);
    
    t1->treenode1 = createTreenode(3,0,NULL,NULL,NULL,NULL);
    t1->treenode1->treenode1 = genprog(prog->maxdepth,prog);//createTreenode(4,3,NULL,NULL,NULL,createExp(0,1));
    //t1->treenode1->treenode1->fixed = 1;
    
    //t1->treenode1->treenode2 = createTreenode(3,0,NULL,NULL,NULL,NULL);
    //t1->treenode1->treenode2->treenode1 = genprog(prog->maxdepth,prog);
    t1->treenode1->treenode2 = createTreenode(1,0,NULL,NULL,NULL,NULL);
    t1->treenode1->treenode2->cond1 = gencond(prog,5);
    t1->treenode1->treenode2->treenode1 = genprog(3,prog);
    t1->treenode1->treenode2->fixed = 1;
    
    t1->treenode2 = createTreenode(3,0,NULL,NULL,NULL,NULL);
    t1->treenode2->treenode1 = createTreenode(5,0,NULL,NULL,NULL,NULL);
    
    
    t1->treenode2->treenode2 = genprog(prog->maxdepth,prog);//createTreenode(4,3,NULL,NULL,NULL,createExp(0,0));
    //t1->treenode2->treenode2->fixed = 1;
    
    //while(assign->exp1->type == 1 && assign->exp1->index == assign->index)
    //    assign->exp1 = genexp(prog,-1);
    
    
    return result;
}

void addCS(program* prog)
{
    treeRank* tr = (treeRank*)malloc(sizeof(treeRank));
    tr->candidate = (mutationNode**)malloc(sizeof(mutationNode*) * 10);
    tr->numcandidate = 0;
    tr->maxnumcandidate = 10;
    tr->ranksum = 0;
    
    searchNode(prog->root,tr,5,prog->maxdepth);
    treenode* node = chooseNode(tr)->node;
    treenode* newnode = genCS(node->depth,prog);
    
    if(node == prog->root)
        prog->root = newnode;
    else
    {
        if(node->parent->treenode1 == node)
            node->parent->treenode1 = newnode;
        else
            node->parent->treenode2 = newnode;
        newnode->parent = node->parent;
        free(node);
    }
}


//0:CONST  1:VAR  2:TIMES  3:PLUS  4:MINUS //5:DIV
void printexp(exp_* e,program* prog)
{
    switch(e->type)
    {
        case 0:    if(e->index < 2)
            printf("%d",e->index);
        else if(e->index == 3)
            printf("me");
        else if(e->index == 4)
            printf("other");
            break;
        case 1:    if(e->index <= 2)
            printf("v[%d]",e->index);
        else if(e->index == 3)
            printf("v[me]");
        else if(e->index == 4)
            printf("v[other]");
            //case 1:printf("v[%d]",e->index);break;
    }
}

//0:TRUE  1:FALSE  2:AND  3:OR  4:NOT
//5:EQ    6:NEQ    7:GEQ  8:GT
void printcond(cond* c,program* prog)
{
    if(c == NULL)
        return;
    switch(c->type)
    {
        case -1:printf("wi");break;
        case 0:printf("true");break;
        case 1:printexp(c->exp1,prog);printf(" == ");printexp(c->exp2,prog);break;
        case 2:printexp(c->exp1,prog);printf(" != ");printexp(c->exp2,prog);break;
        case 3:printf("(");printcond(c->cond1,prog);printf(") && (");printcond(c->cond2,prog);printf(")");break;
        case 4:printf("(");printcond(c->cond1,prog);printf(") || (");printcond(c->cond2,prog);printf(")");break;
    }
}

//0:IF  1:WHILE  2:UNTIL  3:SEQ  4:ASGN
void printprog(treenode* root,int blank,program* prog)
{    //printf("printprog progtype:%d blank:%d\n",prog->type,blank);
    int i;
    if(root == NULL)
        return;
    //printf("parent type :%d",prog->parent->type);
    switch(root->type)
    {
        case 0: for(i = 0;i < blank;i++)printf(" ");printf("if(");
            printcond(root->cond1,prog);
            printf(")\n");
            for(i = 0;i < blank;i++)printf(" ");printf("{\n");
            printprog(root->treenode1,blank + 2,prog);
            for(i = 0;i < blank;i++)printf(" ");printf("}\n");
            if(root->treenode2 == NULL)break;
            for(i = 0;i < blank;i++)printf(" ");printf("else\n");
            for(i = 0;i < blank;i++)printf(" ");printf("{\n");
            printprog(root->treenode2,blank + 2,prog);
            for(i = 0;i < blank;i++)printf(" ");printf("}\n");
            break;
        case 1: for(i = 0;i < blank;i++)printf(" ");printf("while(");
            printcond(root->cond1,prog);
            printf(")\n");
            for(i = 0;i < blank;i++)printf(" ");printf("{\n");
            printprog(root->treenode1,blank + 2,prog);
            for(i = 0;i < blank;i++)printf(" ");printf("}\n");
            break;
        case 2: for(i = 0;i < blank;i++)printf(" ");printf("do\n");
            for(i = 0;i < blank;i++)printf(" ");printf("{\n");
            printprog(root->treenode1,blank + 2,prog);
            for(i = 0;i < blank;i++)printf(" ");printf("}while(");
            printcond(root->cond1,prog);
            printf(");\n");
            break;
        case 3: printprog(root->treenode1,blank,prog);
            printprog(root->treenode2,blank,prog);
            break;
        case 4: for(i = 0;i < blank;i++)printf(" ");
            if(root->index <= 2)
                printf("v[%d] = ",root->index);
            else if(root->index == 3)
                printf("v[me] = ");
            else if(root->index == 4)    //error
                printf("v[other] = ");
            printexp(root->exp1,prog);
            printf(";\n");
            break;
        case 5: for(i = 0;i < blank;i++)printf(" ");
            printf("critical section\n");
    }
}
//setnodenum



//0:IF  1:WHILE  2:UNTIL  3:SEQ  4:ASGN
treenode* findNode(treenode* root, program* prog, int number)
{
    treenode* result = NULL;
    int i;
    if(root == NULL)
        return result;
    //printf("parent type :%d",prog->parent->type);
    switch(root->type)
    {
        case 0:
            if(root->number == number)
                result = root;
            else
                result = findNode(root->treenode1, prog, number);
            break;
            
        case 1:
            if(root->number == number)
                result = root;
            else
                result = findNode(root->treenode1, prog, number);
            break;
            
        case 3:
            result = findNode(root->treenode1, prog, number);
            if (result == NULL) {
                result = findNode(root->treenode2, prog, number);
            }
            break;
            
        case 4:
            if(root->number == number)
                result = root;
            break;
        case 5:
            if(root->number == number)
        {
            printf("wrong number");
            result = NULL;
        }
            
    }
    return result;
}


//0:CONST  1:VAR  2:TIMES  3:PLUS  4:MINUS //5:DIV
void setExpNum(exp_* e,program* prog, int number)
{
    if(e == NULL)
        return;
    
    
    e->number = number;
    
}

//0:TRUE  1:FALSE  2:AND  3:OR  4:NOT
//5:EQ    6:NEQ    7:GEQ  8:GT
int setCondNum(cond* c,program* prog, int number)
{
    if(c == NULL)
        return 0;
    //  printf("condition{");
    //  printf("%d",c->type);
    switch(c->type)
    {
        case -1:c->number = number;number++;break;
        case 0:c->number = number;number++;break;
        case 1:c->number = number;number++;
            setExpNum(c->exp1,prog,number);
            number++;
            setExpNum(c->exp2,prog,number);
            number++;
            break;
        case 2:c->number = number;number++;
            setExpNum(c->exp1,prog,number);number++;
            setExpNum(c->exp2,prog,number);number++;
            break;
        case 3:c->number = number;number++;
            number = setCondNum(c->cond1,prog,number);
            number = setCondNum(c->cond2,prog,number);
            break;
        case 4:c->number = number;number++;
            number = setCondNum(c->cond1,prog,number);
            number = setCondNum(c->cond2,prog,number);
            break;
    }
    //    printf("}");
    return number;
}

void printAst(program* prog)
{
    setTreenodeNum(prog->root, prog, 0);
    char* s = (char*)malloc(sizeof(char)*1000);
    s = newprintprog(prog->root,prog,s);
    s[strlen(s) - 1] = 0;
    FILE *fp;
    fp = fopen("./data/ast.txt","w+");
    fprintf(fp,"%s",s);
    fclose(fp);
}

int setTreenodeNum(treenode* root, program* prog, int number)
{
    if(root == NULL)
        return 0;
    //printf("parent type :%d",prog->parent->type);
    switch(root->type)
    {
        case 0:root->number = number;
            number ++;
            number = setCondNum(root->cond1, prog, number);
            number = setTreenodeNum(root->treenode1, prog, number);
            break;
            
        case 1: root->number = number;
            number ++;
            number = setCondNum(root->cond1, prog, number);
            number = setTreenodeNum(root->treenode1, prog, number);
            break;
            
        case 3: number = setTreenodeNum(root->treenode1,prog,number);
            //printf(" ");
            number = setTreenodeNum(root->treenode2,prog,number);
            break;
            
        case 4: root->number = number;
            number++;
            number++;
            setExpNum(root->exp1, prog, number);
            number++;
            break;
        case 5:root->number = number;
            number++;
    }
    return number;
}



//newprint

//0:CONST  1:VAR  2:TIMES  3:PLUS  4:MINUS //5:DIV
char* newprintexp(exp_* e,program* prog, int node, char* s)
{
    
    if(e == NULL)
        return s;
    if (node == 1) {
        switch(e->type)
        {
            case 0:    if(e->index < 2)
                strcat(s,iToStr(e->index));
            else if(e->index == 3)
                strcat(s,"me");
            else if(e->index == 4)
                strcat(s,"other");
                break;
            case 1:    if(e->index <= 2)
            {
                strcat(s,"v[");
                strcat(s,iToStr(e->index));
                strcat(s,"]");
            }
            else if(e->index == 3)
                strcat(s,"v[me]");
            else if(e->index == 4)
                strcat(s,"v[other]");
                //case 1:strcat(s,"v[%d]",e->index);break;
        }
    }
    else{
        strcat(s, iToStr(e->number));
    }
    return s;
}

//0:CONST  1:VAR  2:TIMES  3:PLUS  4:MINUS //5:DIV
char* newprintexp_(exp_* e,program* prog, char* s)
{
    
    if(e == NULL)
        return s;
    
    //   printf("%d", e->number);
    switch(e->type)
    {
        case 0:    if(e->index < 2)
            strcat(s, iToStr(e->index));
        else if(e->index == 3)
            strcat(s, "me");
        else if(e->index == 4)
            strcat(s, "other");
            break;
        case 1:    if(e->index <= 2)
        {
            strcat(s, "v[");
            strcat(s, iToStr(e->index));
            strcat(s, "]");
        }
        else if(e->index == 3)
            strcat(s, "v[me]");
        else if(e->index == 4)
            strcat(s, "v[other]");
            //case 1:printf("v[%d]",e->index);break;
    }
    return s;
}
//expNode
void expNode(exp_* e,program* prog)
{
    
    if(e == NULL)
        return;
    
    //  printf("expression{");
    printf("%d", e->number);
    switch(e->type)
    {
        case 0:    if(e->index < 2)
            printf("%d",e->index);
        else if(e->index == 3)
            printf("me");
        else if(e->index == 4)
            printf("other");
            break;
        case 1:    if(e->index <= 2)
            printf("v[%d]",e->index);
        else if(e->index == 3)
            printf("v[me]");
        else if(e->index == 4)
            printf("v[other]");
            //case 1:printf("v[%d]",e->index);break;
    }
    //  printf("}");
}


//condNode
char* condNode(cond* c,program* prog, int node, char* s)
{
    //   printf("%d",c->type);
    if(c == NULL)
        printf("nullCond");
    // sprintf(s, "%d", c->number);
    if (node == 0) {
        strcat(s, iToStr(c->number));
        
    }
    else{
        switch(c->type)
        {
            case -1:strcat(s,"wi");break;
            case 0:strcat(s,"true");break;
            case 1:strcat(s,"eq");break;
            case 2:strcat(s,"neq");break;
            case 3:strcat(s,"and");break;
            case 4:strcat(s,"or");break;
        }
    }
    return s;
    
}

char* iToStr(int number)
{
    char* str = (char*)malloc(sizeof(char));
    sprintf(str, "%d", number);
    return str;
}

//0:IF  1:WHILE  2:UNTIL  3:SEQ  4:ASGN
char* newprintprog(treenode* root,program* prog, char* s)
{    //printf("newprintprog progtype:%d blank:%d\n",prog->type,blank);
    if(root == NULL)
        return s;
    //strcat(s,"parent type :%d",prog->parent->type);
    switch(root->type)
    {
        case 0://treenode
            strcat(s,iToStr(root->number));
            strcat(s, "node:if;");
            strcat(s, "children:");
            s = condNode(root->cond1, prog, 0, s);
            strcat(s,";");
            s = treeNode_(root->treenode1, prog, 0, s);
            strcat(s,";\n");
            //condnode
            strcat(s,iToStr(root->cond1->number));
            strcat(s,"node:");
            s = condNode(root->cond1, prog, 1, s);
            strcat(s,";children:");
            //strcat(s,"%d", root->cond1->type);
            if (root->cond1->type < 1)
            {
                strcat(s,"null\n");
            }
            else if(root->cond1->type < 3)
            {
                s = newprintexp(root->cond1->exp1, prog, 0, s);
                strcat(s,";");
                s = newprintexp(root->cond1->exp2, prog, 0, s);
                strcat(s,";\n");
                //exp1
                strcat(s,iToStr(root->cond1->exp1->number));
                strcat(s,"node:");
                s = newprintexp(root->cond1->exp1, prog, 1, s);
                strcat(s,";children:null\n");
                //exp2
                strcat(s,iToStr(root->cond1->exp2->number));
                strcat(s,"node:");
                s = newprintexp(root->cond1->exp2, prog, 1, s);
                strcat(s,";children:null\n");
            }
            else
            {
                s = condNode(root->cond1->cond1, prog, 0, s);
                strcat(s,";");
                s = condNode(root->cond1->cond2, prog, 0, s);
                strcat(s,"\n");
                //cond1
                strcat(s,iToStr(root->cond1->cond1->number));
                strcat(s,"node:");
                s = condNode(root->cond1->cond1, prog, 1, s);
                strcat(s,";");
                strcat(s,"children:");
                s = newprintexp(root->cond1->cond1->exp1, prog, 0, s);
                strcat(s,";");
                s = newprintexp(root->cond1->cond1->exp2, prog, 0, s);
                strcat(s,";\n");
                //exp1
                strcat(s,iToStr(root->cond1->cond1->exp1->number));
                strcat(s,"node:");
                s = newprintexp(root->cond1->cond1->exp1, prog, 1, s);
                strcat(s,";children:null\n");
                //exp2
                strcat(s,iToStr(root->cond1->cond1->exp2->number));
                strcat(s,"node:");
                s = newprintexp(root->cond1->cond1->exp2, prog, 1, s);
                strcat(s,";children:null\n");
                //cond2
                strcat(s,iToStr(root->cond1->cond2->number));
                strcat(s,"node:");
                s = condNode(root->cond1->cond2, prog, 1, s);
                strcat(s,";");
                strcat(s,"children:");
                s = newprintexp(root->cond1->cond2->exp1, prog, 0, s);
                strcat(s,";");
                s = newprintexp(root->cond1->cond2->exp2, prog, 0, s);
                strcat(s,";\n");
                //exp1
                strcat(s,iToStr(root->cond1->cond2->exp1->number));
                strcat(s,"node:");
                s = newprintexp(root->cond1->cond2->exp1, prog, 1, s);
                strcat(s,";children:null\n");
                //exp2
                strcat(s,iToStr(root->cond1->cond2->exp2->number));
                strcat(s,"node:");
                s = newprintexp(root->cond1->cond2->exp2, prog, 1, s);
                strcat(s,";children:null\n");
            }
            
            //  newprintcond(root->cond1, prog);
            s = newprintprog(root->treenode1,prog, s);
            break;
            
        case 1://treenode
            strcat(s ,iToStr(root->number));
            strcat(s, "node:while;");
            strcat(s, "children:");
            s = condNode(root->cond1, prog, 0, s);
            strcat(s, ";");
            s = treeNode_(root->treenode1, prog, 0, s);
            strcat(s, ";\n");
            
            //condnode
            strcat(s, iToStr(root->cond1->number));
            strcat(s, "node:");
            s = condNode(root->cond1, prog, 1, s);
            strcat(s, ";children:");
            //  strcat(s,"%d", root->cond1->type);
            if (root->cond1->type < 1)
            {
                strcat(s,"null\n");
            }
            else if(root->cond1->type < 3)
            {
                s = newprintexp(root->cond1->exp1, prog, 0, s);
                strcat(s,";");
                s = newprintexp(root->cond1->exp2, prog, 0, s);
                strcat(s,";\n");
                //exp1
                strcat(s,iToStr(root->cond1->exp1->number));
                strcat(s,"node:");
                s = newprintexp(root->cond1->exp1, prog, 1, s);
                strcat(s,";children:null\n");
                //exp2
                strcat(s,iToStr(root->cond1->exp2->number));
                strcat(s,"node:");
                s = newprintexp(root->cond1->exp2, prog, 1, s);
                strcat(s,";children:null\n");
            }
            else
            {
                s = condNode(root->cond1->cond1, prog, 0, s);
                strcat(s,";");
                s = condNode(root->cond1->cond2, prog, 0, s);
                strcat(s,"\n");
                //cond1
                strcat(s,iToStr(root->cond1->cond1->number));
                strcat(s,"node:");
                s = condNode(root->cond1->cond1, prog, 1, s);
                strcat(s,";");
                strcat(s,"children:");
                s = newprintexp(root->cond1->cond1->exp1, prog, 0, s);
                strcat(s,";");
                s = newprintexp(root->cond1->cond1->exp2, prog, 0, s);
                strcat(s,";\n");
                //exp1
                strcat(s,iToStr(root->cond1->cond1->exp1->number));
                strcat(s,"node:");
                s = newprintexp(root->cond1->cond1->exp1, prog, 1, s);
                strcat(s,";children:null\n");
                //exp2
                strcat(s,iToStr(root->cond1->cond1->exp2->number));
                strcat(s,"node:");
                s = newprintexp(root->cond1->cond1->exp2, prog, 1, s);
                strcat(s,";children:null\n");
                //cond2
                strcat(s,iToStr(root->cond1->cond2->number));
                strcat(s,"node:");
                s = condNode(root->cond1->cond2, prog, 1, s);
                strcat(s,";");
                strcat(s,"children:");
                s = newprintexp(root->cond1->cond2->exp1, prog, 0, s);
                strcat(s,";");
                s = newprintexp(root->cond1->cond2->exp2, prog, 0, s);
                strcat(s,";\n");
                //exp1
                strcat(s,iToStr(root->cond1->cond2->exp1->number));
                strcat(s,"node:");
                s = newprintexp(root->cond1->cond2->exp1, prog, 1, s);
                strcat(s,";children:null\n");
                //exp2
                strcat(s,iToStr(root->cond1->cond2->exp2->number));
                strcat(s,"node:");
                s = newprintexp(root->cond1->cond2->exp2, prog, 1, s);
                strcat(s,";children:null\n");
            }
            //  newprintcond(root->cond1, prog);
            s = newprintprog(root->treenode1, prog, s);
            break;
            
        case 3: s = newprintprog(root->treenode1,prog, s);
            //strcat(s," ");
            s = newprintprog(root->treenode2, prog, s);
            break;
            
        case 4:strcat(s,iToStr(root->number));
            strcat(s,"node:assign;");
            strcat(s,"children:");
            
            int number = root->number;
            strcat(s,iToStr(++number));
            strcat(s,";");
            s = newprintexp(root->exp1, prog,false, s);
            strcat(s,"\n");
            //exp1
            strcat(s,iToStr(number));
            strcat(s,"node:");
            if(root->index <= 2)
            {
                strcat(s, "v[");
                strcat(s, iToStr(root->index));
                strcat(s, "]");
                
            }
            else if(root->index == 3)
            {
                strcat(s, "v[me]");
            }
            strcat(s, ";children:null\n");
            //exp2
            strcat(s,iToStr(++number));
            strcat(s, "node:");
            s = newprintexp_(root->exp1, prog, s);
            strcat(s, ";children:null\n");
            break;
        case 5:strcat(s,iToStr(root->number));
            strcat(s, "node:cs;children:null\n");
    }
    return s;
}

//treeNode
char* treeNode_(treenode* root,program* prog, int node, char* s)
{    //sprintf(s,"newprintprog progtype:%d blank:%d\n",prog->type,blank);
    //  int i;
    if(root == NULL)
        return s;
    //sprintf(s,"parent type :%d",prog->parent->type);
    if (node == 0 && root->type != 3) {
        strcat(s,iToStr(root->number));
    }
    else{
        switch(root->type)
        {
            case 0://treenode
                strcat(s,iToStr(root->number));
                strcat(s,"if");
                break;
                
            case 1:strcat(s,iToStr(root->number));
                strcat(s,"while");
                break;
                
            case 3:s = treeNode_(root->treenode1,prog, 0, s);
                strcat(s,";");
                s = treeNode_(root->treenode2,prog, 0, s);
                break;
                
            case 4:strcat(s,iToStr(root->number));
                strcat(s,"assign");
                break;
                
            case 5:strcat(s,iToStr(root->number));
                strcat(s,"cs");
        }
    }
    return s;
}

int setNumOfStatementsUp(treenode* t)
{
    if(t == NULL)
        return 0;
    
    int l = setNumOfStatementsUp(t->treenode1);
    int r = setNumOfStatementsUp(t->treenode2);
    
    if(t->type != 3)
        t->numofstatements = 1;
    else
        t->numofstatements = l + r;
    return t->numofstatements;
}

void setNumOfStatementsDown(treenode* t,int num)
{
    if(t == NULL)
        return;
    if(t->type == 3)
    {
        if(num > t->numofstatements)
            t->numofstatements = num;
        
        setNumOfStatementsDown(t->treenode1,t->numofstatements);
        setNumOfStatementsDown(t->treenode2,t->numofstatements);
    }
    else
    {
        t->numofstatements = num;
        setNumOfStatementsDown(t->treenode1,1);
        setNumOfStatementsDown(t->treenode2,1);
    }
}

void setNumOfStatements(treenode* root)
{
    setNumOfStatementsUp(root);
    setNumOfStatementsDown(root,1);
}

void setLinesTreenode(treenode* t,int depth)
{
    if(t == NULL)
        return;
    //printf("1.0\n");
    
    t->depth = depth;
    
    //printf("1.5:%d\n",t->type);
    //if(t->parent != NULL)
    //    printf("1.6:%d\n",t->parent->type);
    if(t->type == 3)
    {
        setLinesTreenode(t->treenode1,depth);
        setLinesTreenode(t->treenode2,depth);
    }
    else
    {
        setLinesTreenode(t->treenode1,depth + 1);
        setLinesTreenode(t->treenode2,depth + 1);
    }
    //printf("2\n");
    int height = 0;
    if(t->treenode1 != NULL)
        if(t->treenode1->height > height)
            height = t->treenode1->height;
    if(t->treenode2 != NULL)
        if(t->treenode2->height > height)
            height = t->treenode2->height;
    //printf("3\n");
    if(t->type == 3)
        t->height = height;
    else
        t->height = height + 1;
}

int setFixed(treenode* t)
{
    if(t == NULL)
        return 0;
    
    if(t->fixed == 1)
    {
        setFixed(t->treenode1);
        setFixed(t->treenode2);
        return 1;
    }
    
    if(t->type == 0)
    {
        int a = setFixed(t->treenode1);
        int b = setFixed(t->treenode2);
        if(a == 0 && b == 0)
            t->fixed = 0;
        else
            t->fixed = 1;
    }
    else if(t->type == 1)
        t->fixed = setFixed(t->treenode1);
    else if(t->type == 2)
        t->fixed = setFixed(t->treenode1);
    else if(t->type == 3)
    {    if(t->treenode1 == NULL || t->treenode2 == NULL)
        printf("setfixed error\n");
        int a = setFixed(t->treenode1);
        int b = setFixed(t->treenode2);
        if(a == 0 && b == 0)
            t->fixed = 0;
        else
            t->fixed = 1;
    }
    else if(t->type == 4)
        t->fixed = 0;
    else if(t->type == 5)
        t->fixed = 1;
    
    return t->fixed;
}

void printType(treenode* t)
{
    if(t == NULL)
        return;
    //if(t->next == NULL)
    printf("%d,%d,%d,%d; ",t->type,t->depth,t->fixed,t->numofstatements);
    //else
    //    printf("%d,%d,%d,%d; ",t->type,t->depth,t->fixed,t->next->type);
    printType(t->treenode1);
    printType(t->treenode2);
}
int existCS(treenode* t)
{
    if(t == NULL)
        return 0;
    if(t->type == 5)
        return 1;
    else if(existCS(t->treenode1) == 1 || existCS(t->treenode2) == 1)
        return 1;
    return 0;
    
}

void setAll(program* prog)
{
    
    //printf("x0\n");
    //printType(org->progs[i]->root);printf("x0.5\n");
    prog->root->next = NULL;
    setNext(prog->root);//printf("x1\n");
    setParent(prog->root);//printf("x2\n");
    setFixed(prog->root);//printf("x3\n");
    setNumOfStatements(prog->root);//printf("x4\n");
    setLinesTreenode(prog->root,1);//printf("x5\n");
    
}

int getFixed(treenode* t)
{
    if(t == NULL)
        return 0;
    else
        return t->fixed;
}

program** genInitTemplate(int num)
{
    program** inittemplate = (program**)malloc(sizeof(program*) * num);
    int i;
    for(i = 0;i < num;i++)
    {
        inittemplate[i] = (program*)malloc(sizeof(program));
        inittemplate[i]->maxdepth = 4;
        //inittemplate[i]->progid = j;
        inittemplate[i]->maxconst = maxconst;
        inittemplate[i]->numprivatevars = numprivatevars;
        inittemplate[i]->numpublicvars = numpublicvars;
        inittemplate[i]->root = genCS(1,inittemplate[i]);
        inittemplate[i]->checkedBySpin = 0;
        setAll(inittemplate[i]);
        
    }
    return inittemplate;
}

void setMeOther(treenode* t,cond* c,exp_* e,int type,int progid)
{
    if(type == 1)
    {
        if(t == NULL)
            return;
        if(t->type == 4 && t->index >= 3)
        {
            if(progid == 0)    //3me:0    4other:1
                t->index = t->index - 3;
            else if(progid == 1)    //3me:1    4other:0
                t->index = 4 - t->index;
            else
                printf("setmeother error progid != 0,1\n");
        }
        setMeOther(t->treenode1,NULL,NULL,1,progid);
        setMeOther(t->treenode2,NULL,NULL,1,progid);
        setMeOther(NULL,t->cond1,NULL,2,progid);
        setMeOther(NULL,NULL,t->exp1,3,progid);
    }
    else if(type == 2)
    {
        if(c == NULL)
            return;
        if(c->type != 0)
        {
            setMeOther(NULL,NULL,c->exp1,3,progid);
            setMeOther(NULL,NULL,c->exp2,3,progid);
            setMeOther(NULL,c->cond1,NULL,2,progid);
            setMeOther(NULL,c->cond2,NULL,2,progid);
        }
    }
    else
    {
        if(e == NULL)
            return;
        if(e->index >= 3)
        {
            if(progid == 0)    //3me:0    4other:1
                e->index = e->index - 3;
            else if(progid == 1)    //3me:1    4other:0
                e->index = 4 - e->index;
            else
                printf("setmeother error progid != 0,1\n");
        }
    }
}

program* genProgram(program* templat,int progid)
{
    program* newprog = copyProgram(templat);
    newprog->progid = progid;
    setMeOther(newprog->root,NULL,NULL,1,progid);
    
    setAll(newprog);
    return newprog;
}

organism* genOrganism(program* templat)
{
    organism* result = (organism*)malloc(sizeof(organism));
    result->progs = (program**)malloc(sizeof(program*) * 2);
    result->progs[0] = genProgram(templat,0);
    result->progs[1] = genProgram(templat,1);
    if(existNullCond(result->progs[0]->root))
    {
        if(!existNullCond(templat->root))
            printf("organism generate ERROR!!\n\n");
        else
        {
            printf("Template\n");
            printprog(templat->root,0,templat);
            printf("program0\n");
            printprog(result->progs[0]->root,0,result->progs[0]);
            printf("program1\n");
            printprog(result->progs[1]->root,0,result->progs[1]);
            printf("\n");
        }
    }
    return result;
}

void freeAll(organism* org,program* prog,treenode* t,cond* c,exp_* e,int type)
{
    switch(type)
    {
        case 1:    if(org == NULL)
            break;
            freeAll(NULL,org->progs[0],NULL,NULL,NULL,2);
            freeAll(NULL,org->progs[1],NULL,NULL,NULL,2);
            free(org);
            break;
        case 2:    if(prog == NULL)
            return;
            freeAll(NULL,NULL,prog->root,NULL,NULL,3);
            free(prog);
            break;
        case 3:    if(t == NULL)
            break;
            freeAll(NULL,NULL,t->treenode1,NULL,NULL,3);
            freeAll(NULL,NULL,t->treenode2,NULL,NULL,3);
            freeAll(NULL,NULL,NULL,t->cond1,NULL,4);
            freeAll(NULL,NULL,NULL,NULL,t->exp1,5);
            free(t);
            break;
        case 4:    if(c == NULL)
            break;
            freeAll(NULL,NULL,NULL,NULL,c->exp1,5);
            freeAll(NULL,NULL,NULL,NULL,c->exp2,5);
            freeAll(NULL,NULL,NULL,c->cond1,NULL,4);
            freeAll(NULL,NULL,NULL,c->cond2,NULL,4);
            free(c);
            break;
        case 5:    if(e == NULL)
            break;
            free(e);
            break;
    }
}


void expToPml(exp_* e,FILE* f)
{
    if(e == NULL)
        return;
    
    if(e->type == 0)
    {
        if(e->index == 0)
            fputs("0",f);
        else if(e->index == 1)
            fputs("1",f);
    }
    else if(e->type == 1)
    {
        if(e->index == 0)
            fputs("v0",f);
        else if(e->index == 1)
            fputs("v1",f);
        else if(e->index == 2)
            fputs("turn",f);
    }
}
void condToPml(cond* c,FILE* f,int blank, int progid)
{
    if(c == NULL)
        return;
    int i;
    for(i = 0;i < blank;i++)
        fputs("\t",f);
    fputs("::",f);
    
    if(c->type == -1)
    {
        if(progid == 0)
        {
            fputs("wi0 == 0->\n",f);
            for(i = 0;i < blank + 1;i++)
                fputs("\t",f);
            fputs("try0 = 1;\n",f);
            
            for(i = 0;i < blank + 1;i++)
                fputs("\t",f);
            fputs("select(wi0:0..1);\n",f);
        }
        else if(progid == 1)
        {
            fputs("wi1 == 0->\n",f);
            for(i = 0;i < blank + 1;i++)
                fputs("\t",f);
            fputs("try1 = 1;\n",f);
            
            for(i = 0;i < blank + 1;i++)
                fputs("\t",f);
            fputs("select(wi1:0..1);\n",f);
        }
    }
    else if(c->type == 0)
        fputs("true->\n",f);
    else if(c->type == 1)
    {
        expToPml(c->exp1,f);
        fputs(" == ",f);
        expToPml(c->exp2,f);
        fputs("->\n",f);
    }
    else if(c->type == 2)
    {
        expToPml(c->exp1,f);
        fputs(" != ",f);
        expToPml(c->exp2,f);
        fputs("->\n",f);
    }
    else if(c->type == 3)
    {
        if(c->cond1->type == 0)
            fputs("true",f);
        else if(c->cond1->type == 1)
        {
            expToPml(c->cond1->exp1,f);
            fputs(" == ",f);
            expToPml(c->cond1->exp2,f);
        }
        else if(c->cond1->type == 2)
        {
            expToPml(c->cond1->exp1,f);
            fputs(" != ",f);
            expToPml(c->cond1->exp2,f);
        }
        
        fputs(" && ",f);
        
        if(c->cond2->type == 0)
            fputs("true",f);
        else if(c->cond2->type == 1)
        {
            expToPml(c->cond2->exp1,f);
            fputs(" == ",f);
            expToPml(c->cond2->exp2,f);
        }
        else if(c->cond2->type == 2)
        {
            expToPml(c->cond2->exp1,f);
            fputs(" != ",f);
            expToPml(c->cond2->exp2,f);
        }
        fputs("->\n",f);
    }
    else if(c->type == 4)
    {
        if(c->cond1->type == 0)
            fputs("true",f);
        else if(c->cond1->type == 1)
        {
            expToPml(c->cond1->exp1,f);
            fputs(" == ",f);
            expToPml(c->cond1->exp2,f);
        }
        else if(c->cond1->type == 2)
        {
            expToPml(c->cond1->exp1,f);
            fputs(" != ",f);
            expToPml(c->cond1->exp2,f);
        }
        
        fputs(" || ",f);
        
        if(c->cond2->type == 0)
            fputs("true",f);
        else if(c->cond2->type == 1)
        {
            expToPml(c->cond2->exp1,f);
            fputs(" == ",f);
            expToPml(c->cond2->exp2,f);
        }
        else if(c->cond2->type == 2)
        {
            expToPml(c->cond2->exp1,f);
            fputs(" != ",f);
            expToPml(c->cond2->exp2,f);
        }
        fputs("->\n",f);
    }
}

void progToPml(treenode* t,FILE* f,program* prog,int blank)
{
    if(t == NULL)
        return;
    int i;
    if(t->type == 0)
    {
        for(i = 0;i < blank;i++)
            fputs("\t",f);
        fputs("if\n",f);
        condToPml(t->cond1,f,blank + 1,prog->progid);
        progToPml(t->treenode1,f,prog,blank + 2);
        
        for(i = 0;i < blank;i++)
            fputs("\t",f);
        fputs("fi\n",f);
    }
    else if(t->type == 1)
    {
        for(i = 0;i < blank;i++)
            fputs("\t",f);
        fputs("do\n",f);
        condToPml(t->cond1,f,blank + 1,prog->progid);
        progToPml(t->treenode1,f,prog,blank + 2);
        
        for(i = 0;i < blank + 1;i++)
            fputs("\t",f);
        fputs("::else->break;\n",f);
        
        for(i = 0;i < blank;i++)
            fputs("\t",f);
        fputs("od\n",f);
    }
    else if(t->type == 3)
    {
        progToPml(t->treenode1,f,prog,blank);
        progToPml(t->treenode2,f,prog,blank);
    }
    else if(t->type == 4)
    {
        for(i = 0;i < blank;i++)
            fputs("\t",f);
        if(t->index == 0)
            fputs("v0 = ",f);
        else if(t->index == 1)
            fputs("v1 = ",f);
        else if(t->index == 2)
            fputs("turn = ",f);
        
        expToPml(t->exp1,f);
        fputs(";\n",f);
    }
    else if(t->type == 5)
    {
        if(prog->progid == 0)
        {
            for(i = 0;i < blank;i++)
                fputs("\t",f);
            fputs("cs0++;\n",f);
            for(i = 0;i < blank;i++)
                fputs("\t",f);
            fputs("try0 = 0;\n",f);
            for(i = 0;i < blank;i++)
                fputs("\t",f);
            fputs("cs0--;\n",f);
        }
        else if(prog->progid == 1)
        {
            for(i = 0;i < blank;i++)
                fputs("\t",f);
            fputs("cs1++;\n",f);
            for(i = 0;i < blank;i++)
                fputs("\t",f);
            fputs("try1 = 0;\n",f);
            for(i = 0;i < blank;i++)
                fputs("\t",f);
            fputs("cs1--;\n",f);
        }
    }
}

void orgToPml(organism* org,FILE* f)
{
    fputs("bit turn = 0,v0 = 0,v1 = 0,try0 = 0,try1 = 0,wi0 = 0,wi1 = 0;\nbyte cs0 = 0,cs1 = 0;\n#define mutex (cs0 + cs1 <= 1)\n\n",f);
    fputs("active proctype p()\n{\n\tselect(wi0:0..1);\n",f);
    progToPml(org->progs[0]->root,f,org->progs[0],1);
    fputs("}\n\n",f);
    fputs("active proctype q()\n{\n\tselect(wi1:0..1);\n",f);
    progToPml(org->progs[1]->root,f,org->progs[1],1);
    fputs("}\n\nltl e1{[]mutex}\nltl e2{[]((try0 == 1) -> <>(cs0 == 1))}\nltl e3{[]((try1 == 1) -> <>(cs1 == 1))}",f);
}

int compareExp(exp_* e1,exp_* e2)
{
    if(e1 == NULL && e2 == NULL)
        return 1;
    if(e1 == NULL || e2 == NULL)
        return 0;
    
    if(e1->type == e2->type && e1->index == e2->index)
        return 1;
    return 0;
}
int compareCond(cond* c1,cond* c2)
{
    if(c1 == NULL && c2 == NULL)
        return 1;
    if(c1 == NULL || c2 == NULL)
        return 0;
    
    if(c1->type == c2->type)
    {
        if(c1->type == 0)
            return 1;
        else if(c1->type == 1 || c1->type == 2)
        {
            if(compareExp(c1->exp1,c2->exp1) == 1 && compareExp(c1->exp2,c2->exp2) == 1)
                return 1;
            else if(compareExp(c1->exp1,c2->exp2) == 1 && compareExp(c1->exp2,c2->exp1) == 1)
                return 1;
        }
        else if(c1->type == 3 || c1->type == 4)
        {
            if(compareCond(c1->cond1,c2->cond1) == 1 && compareCond(c1->cond2,c2->cond2) == 1)
                return 1;
            else if(compareCond(c1->cond1,c2->cond2) == 1 && compareCond(c1->cond2,c2->cond1) == 1)
                return 1;
        }
    }
    return 0;
}

int compareTreenode(treenode* t1,treenode* t2)
{
    if(t1 == NULL && t2 == NULL)
        return 1;
    if(t1 == NULL || t2 == NULL)
        return 0;
    
    if(t1->type == t2->type && t1->index == t2->index && compareExp(t1->exp1,t2->exp1) == 1)
        if(compareCond(t1->cond1,t2->cond1) == 1 && compareTreenode(t1->treenode1,t2->treenode1) == 1 && compareTreenode(t1->treenode2,t2->treenode2) == 1)
            return 1;
    return 0;
}

treenode* example_e()
{
    treenode* result = createTreenode(3,0,NULL,NULL,NULL,NULL);
    result->treenode1 = createTreenode(3,0,NULL,NULL,NULL,NULL);
    result->treenode2 = createTreenode(3,0,NULL,NULL,NULL,NULL);
    result->treenode1->treenode1 = createTreenode(4,3,NULL,NULL,NULL,createExp(0,3));
    
    result->treenode1->treenode2 = createTreenode(0,0,NULL,NULL,NULL,NULL);
    treenode* p = result->treenode1->treenode2;
    p->cond1 = createCond(1,createExp(1,2),createExp(0,3),NULL,NULL);
    
    p->treenode1 = createTreenode(3,0,NULL,NULL,NULL,NULL);
    p->treenode1->treenode1 = createTreenode(4,2,NULL,NULL,NULL,createExp(0,4));
    p->treenode1->treenode2 = createTreenode(1,0,createCond(2,createExp(1,4),createExp(1,2),NULL,NULL),NULL,NULL,NULL);
    
    result->treenode2->treenode1 = createTreenode(5,0,NULL,NULL,NULL,NULL);
    result->treenode2->treenode2 = createTreenode(4,3,NULL,NULL,NULL,createExp(0,4));
    
    treenode* output = createTreenode(1,0,NULL,result,NULL,NULL);
    output->cond1 = createCond(-1,NULL,NULL,NULL,NULL);
    return output;
}



//mutation.c
int mutype;
exp_* copyExp(exp_* e)
{
    if(e == NULL)
        return NULL;
    exp_* result = (exp_*)malloc(sizeof(exp_));
    result->type = e->type;
    result->index = e->index;
    result->number = e->number;
    return result;
}

cond* copyCond(cond* c)
{
    if(c == NULL)
        return NULL;
    cond* result = (cond*)malloc(sizeof(cond));
    result->type = c->type;
    result->exp1 = copyExp(c->exp1);
    result->exp2 = copyExp(c->exp2);
    result->cond1 = copyCond(c->cond1);
    result->cond2 = copyCond(c->cond2);
    result->number = c->number;
    return result;
}

treenode* copyTreenode(treenode* t)
{
    if(t == NULL)
        return NULL;
    treenode* result = (treenode*)malloc(sizeof(treenode));
    result->type = t->type;
    result->index = t->index;
    result->cond1 = copyCond(t->cond1);
    result->treenode1 = copyTreenode(t->treenode1);
    result->treenode2 = copyTreenode(t->treenode2);
    result->exp1 = copyExp(t->exp1);
    result->goodexamples = t->goodexamples;
    result->badexamples = t->badexamples;
    result->depth = t->depth;
    result->height = t->height;
    result->fixed = t->fixed;
    result->numofstatements = t->numofstatements;        //!!
    result->number = t->number;
    setParent(result);
    return result;
}

program* copyProgram(program* prog)
{
    program* result = (program*)malloc(sizeof(program));
    result->root = copyTreenode(prog->root);
    result->maxdepth = prog->maxdepth;
    result->progid = prog->progid;
    result->maxconst = prog->maxconst;
    result->numprivatevars = prog->numprivatevars;
    result->numpublicvars = prog->numpublicvars;
    result->checkedBySpin = prog->checkedBySpin;
    return result;
}


int satisfyMutationReduction(treenode* t)        //!!!
{
    if(t->treenode1 == NULL && t->treenode2 == NULL)
        return 0;
    if(t->type == 0)
    {
        if(getFixed(t->treenode1) == 1 || getFixed(t->treenode2) == 1)
            printf("if child fixed error!!\n");
        return 1;
    }
    else if(t->type == 1)// || t->type == 2)
    {
        if(t->fixed == 0 && t->treenode1 != NULL)
            return 1;
    }
    else if(t->type == 3)
    {
        if(getFixed(t->treenode1) == 0 && getFixed(t->treenode2) == 0)    //
            return 1;
    }
    return 0;
}


void mutationCond(cond* root,program* prog,int type)    //type == 1:can add        type == 0:can't add
{
    if(root == NULL)
        return;
    if(root->type == -1)
        return;
    if(root->type == 0)
    {    //printf("0");
        cond* new = gencond(prog,3);
        while(new->type == 0)
        {
            free(new);
            new = gencond(prog,3);
        }
        root->type = new->type;
        root->exp1 = new->exp1;
        root->exp2 = new->exp2;
        root->cond1 = new->cond1;
        root->cond2 = new->cond2;
        free(new);
    }
    else if(root->type == 1 || root->type == 2)
    {
        int t = nextrand(4 + type);//printf("a%d",t);
        if(t == 0)        //change left
            root->exp1 = genexp(prog,1);
        else if(t == 1)        //change right
            root->exp2 = genexp(prog,0);
        else if(t == 2)        //change == / !=
            root->type = 3 - root->type;
        else if(t == 3)                //change entirely
        {
            free(root->exp1);
            free(root->exp2);
            free(root->cond1);
            free(root->cond2);
            
            cond* new;
            if(type == 1)
                new = gencond(prog,5);
            else
                new = gencond(prog,3);
            
            root->type = new->type;
            root->exp1 = new->exp1;
            root->exp2 = new->exp2;
            root->cond1 = new->cond1;
            root->cond2 = new->cond2;
            free(new);
        }
        else         //change to &&/||
        {
            cond* c1 = copyCond(root);
            free(root->exp1);
            root->exp1 = NULL;
            free(root->exp2);
            root->exp2 = NULL;
            root->type = nextrand(2) + 3;
            root->cond1 = c1;
            root->cond2 = gencond(prog,3);
        }
        
    }
    else if(root->type == 3 || root->type == 4)
    {
        int t = nextrand(6);//printf("b%d",t);
        if(t == 0)        //change left
            mutationCond(root->cond1,prog,0);
        else if(t == 1)    //change && / ||
            root->type = 7 - root->type;
        else if(t == 2)    //change right
            mutationCond(root->cond2,prog,0);
        else if(t == 3)    //reduction  left
        {
            root->type = root->cond1->type;
            root->exp1 = root->cond1->exp1;
            root->exp2 = root->cond1->exp2;
            free(root->cond1);
            root->cond1 = NULL;
            free(root->cond2);
            root->cond2 = NULL;
        }
        else if(t == 4)    //reduction right
        {
            root->type = root->cond2->type;
            root->exp1 = root->cond2->exp1;
            root->exp2 = root->cond2->exp2;
            free(root->cond1);
            root->cond1 = NULL;
            free(root->cond2);
            root->cond2 = NULL;
        }
        else if(t == 5)                //change entirely
        {
            free(root->exp1);
            free(root->exp2);
            free(root->cond1);
            free(root->cond2);
            
            cond* new = gencond(prog,5);
            
            root->type = new->type;
            root->exp1 = new->exp1;
            root->exp2 = new->exp2;
            root->cond1 = new->cond1;
            root->cond2 = new->cond2;
            free(new);
        }
    }
}

/*void mutationExp(exp* root,program* prog)
 {
 if(root == NULL)
 return;
 exp* new = genexp(prog,0);
 while(equalExp(root,new) == 1)
 {
 free(new);
 new = genexp(prog,0);
 }
 root->type = new->type;
 root->index = new->index;
 }*/

treenode* getStatement(treenode* seq, int t)    //t=0:first    t=1:last
{
    if(seq == NULL || seq->type != 3)
        return seq;
    
    if(t == 0)
        return getStatement(seq->treenode1,0);
    else
        return getStatement(seq->treenode2,1);
}

program* mutation(program* parent)
{
    program* newprog = copyProgram(parent);
    newprog->checkedBySpin = 0;
    treenode* new = newprog->root;
    
    //printf("mutationtype = %d\n",mutationtype);
    int mutationtype;
    mutationNode* chnode;
    
    do
    {
        treeRank* tr = (treeRank*)malloc(sizeof(treeRank));
        tr->candidate = (mutationNode**)malloc(sizeof(mutationNode*) * 10);
        tr->numcandidate = 0;
        tr->maxnumcandidate = 10;
        tr->ranksum = 0;
        mutationtype = nextrand(4) + 1;
        searchNode(new,tr,mutationtype,newprog->maxdepth);
        chnode = chooseNode(tr);
    }while(chnode == NULL);
    
    mutype = mutationtype;
    if(mutationtype == 1)                            //Replacement Mutation type
    {
        treenode* mnode = chnode->node;
        free(chnode);
        if(mnode->fixed == 0 && (mnode->cond1 == NULL && mnode->exp1 == NULL || nextrand(2) == 0))
        {
            treenode* newnode;
            if(mnode->depth != 2)
                newnode = genprog(mnode->depth,newprog);
            else
                newnode = genprog(newprog->maxdepth,newprog);
            
            if(mnode->parent == NULL)
                new = newnode;
            else
            {
                if(mnode->parent->treenode1 == mnode)
                    mnode->parent->treenode1 = newnode;
                else
                    mnode->parent->treenode2 = newnode;
                newnode->parent = mnode->parent;
                free(mnode);
            }
        }
        else
        {
            if(mnode->exp1 == NULL || (mnode->cond1 != NULL && nextrand(2) == 0))
                mutationCond(mnode->cond1,newprog,1);
            else
            {
                if(nextrand(2) == 0)
                {
                    exp_* e;
                    do
                    {
                        e = genexp(newprog,0);
                    }while(equalExp(e,mnode->exp1) == 1);
                    free(mnode->exp1);
                    mnode->exp1 = e;
                    
                }
                else
                {
                    mnode->index = 5 - mnode->index;    //5 = 2 + me
                }
            }
        }
    }
    else if(mutationtype == 2)                        //Insert Mutation types
    {    //printf("mutation222\n");
        
        treenode* mnode = chnode->node;
        free(chnode);
        //printf("2 mnode depth:%d,type%d\n",mnode->depth,mnode->type);
        //printf("mutation2233\n");
        //int t = nextrand(4);
        int t = nextrand(3);
        if(t == 2)
            t = 3;
        if(mnode->fixed == 1 || mnode->depth + mnode->height == newprog->maxdepth + 1 || mnode->depth == 2)
            t = 3;
        //printf("2 mnode depth:%d,type:%d,t=%d fixed:%d\n",mnode->depth,mnode->type,t,mnode->fixed);
        
        treenode* newnode = createTreenode(t,0,NULL,NULL,NULL,NULL);
        
        if(t == 0 || t == 1)    //if,while
        {
            newnode->cond1 = gencond(newprog,5);
            newnode->treenode1 = mnode;
            if(mnode->parent == NULL)
            {    //printf("first\n");
                mnode->parent = newnode;
                new = newnode;
            }
            else
            {   // treenode* pp =  mnode->parent;
                //printf("second:parent type:%d\n",mnode->parent->type);
                if(mnode->parent->treenode1 == mnode)
                    mnode->parent->treenode1 = newnode;
                else if(mnode->parent->treenode2 == mnode)
                    mnode->parent->treenode2 = newnode;
                else
                    printf("mutation type2 error!\n");
                //printf("1pptype:%d\n",pp->type);
                //printf("1ppdepth:%d\n",pp->depth);
                newnode->parent = mnode->parent;
                mnode->parent = newnode;
                //printf("before\n");
                //printf("2pptype:%d\n",pp->type);
                //printf("2ppdepth:%d\n",pp->depth);
                //printType(pp);
                //printf("after\n");
            }
        }
        else
        {
            int p;
            p = nextrand(2);
            /*
             if(getStatement(mnode,1)->type == 1)
             p = 1;
             else if(getStatement(mnode,0)->type == 5)
             p = 0;
             else
             p = nextrand(2);
             */
            
            if(p == 0)
            {
                newnode->treenode1 = mnode;
                if(mnode->depth == 2)
                    newnode->treenode2 = genprog(newprog->maxdepth,newprog);
                else
                    newnode->treenode2 = genprog(mnode->depth,newprog);
            }
            else
            {
                newnode->treenode2 = mnode;
                if(mnode->depth == 2)
                    newnode->treenode1 = genprog(newprog->maxdepth,newprog);
                else
                    newnode->treenode1 = genprog(mnode->depth,newprog);
            }
            
            if(mnode->parent == NULL)
            {
                mnode->parent = newnode;
                new = newnode;
            }
            else
            {
                if(mnode->parent->treenode1 == mnode)
                    mnode->parent->treenode1 = newnode;
                else
                    mnode->parent->treenode2 = newnode;
                newnode->parent = mnode->parent;
                mnode->parent = newnode;
            }
        }
        //printf("newnodetype%d\n",newnode->type);
    }
    else if(mutationtype == 3)                        //Reduction Mutation type
    {
        treenode* mnode = chnode->node;
        free(chnode);
        int child = nextrand(2);
        if(mnode->treenode1->fixed == 1 || mnode->treenode2 == NULL || child == 0 && mnode->treenode2->fixed == 0)//treenode1 replace mnode
        {
            if(mnode->parent == NULL)    //mnode == new
            {
                new = mnode->treenode1;
                new->parent = NULL;
            }
            else                         //mnode != new
            {
                mnode->treenode1->parent = mnode->parent;
                if(mnode->parent->treenode1 == mnode)
                    mnode->parent->treenode1 = mnode->treenode1;
                else
                    mnode->parent->treenode2 = mnode->treenode1;
            }
            free(mnode);
        }
        else
        {
            if(mnode->parent == NULL)    //mnode == new
            {
                new = mnode->treenode2;
                new->parent = NULL;
            }
            else                         //mnode != new
            {
                mnode->treenode2->parent = mnode->parent;
                if(mnode->parent->treenode1 == mnode)
                    mnode->parent->treenode1 = mnode->treenode2;
                else
                    mnode->parent->treenode2 = mnode->treenode2;
            }
            free(mnode);
        }
    }
    
    else                            //Deletion mutation type
    {
        treenode* mnode = chnode->node;
        free(chnode);
        //printf("mnode depth:%d,mnode type%d\n",mnode->depth,mnode->type);
        if(mnode->treenode1 == NULL)
            mnode->treenode1 = genprog(mnode->depth + 1,newprog);
        else
        {
            free(mnode->treenode1);
            mnode->treenode1 = NULL;
        }
    }
    newprog->root = new;
    return newprog;
}

bool statementsTooLong(treenode* t)
{
    if(t == NULL)
        return false;
    if(t->numofstatements > 3)
        return true;
    if(statementsTooLong(t->treenode1))
        return true;
    if(statementsTooLong(t->treenode1))
        return true;
    return false;
}

int numOfWi(treenode* t)
{
    if(t == NULL)
        return 0;
    int a = 0,b = 0,c = 0;
    if(t->type == 1 && t->cond1->type == -1)
        a = 1;
    b = numOfWi(t->treenode1);
    c = numOfWi(t->treenode2);
    
    return (a + b + c);
}


bool existNullCond(treenode* t)
{
    if(t == NULL)
        return false;
    if(t->type == 0 || t->type == 1)
        if(t->cond1 == NULL || (t->cond1->type == 3 || t->cond1->type == 4) && (t->cond1->cond1 == NULL || t->cond1->cond2 == NULL))
            return true;
    return (existNullCond(t->treenode1) || existNullCond(t->treenode2));
}

bool existMultiCond(treenode* t)
{
    if(t == NULL)
        return false;
    if(t->type == 0 || t->type == 1)
    {
        if(t->cond1->cond1 != NULL && (t->cond1->cond1->type == 3 || t->cond1->cond1->type == 4) )
            return true;
        if(t->cond1->cond2 != NULL && (t->cond1->cond2->type == 3 || t->cond1->cond2->type == 4))
            return true;
    }
    return (existMultiCond(t->treenode1) || existMultiCond(t->treenode2));
}

program** genNewCandidate(int numofcandidate,program** candidate,int numofmutation)
{
    int* selected = (int*)malloc(sizeof(int) * numofcandidate);
    program** result = (program**)malloc(sizeof(program*) * numofmutation);
    int i;
    for(i = 0;i < numofcandidate;i++)
        selected[i] = 0;
    
    int count = 0;
    while(count < numofmutation)
    {
        int index;
        do
        {
            index = nextrand(numofcandidate);
        }while(selected[index] == 1);
        
        selected[index] = 1;
        count++;
    }
    //printf("Choose candidate:");
    //for(i = 0;i < numofcandidate;i++)
    //    printf("%d:%d;     ",i,selected[i]);
    //printf("\n");
    count = 0;
    for(i = 0;i < numofcandidate;i++)
    {
        if(selected[i] == 1)
        {
            //!!result[count] = (program*)malloc(sizeof(program));
            
            
            if(candidate[i] == NULL)printf("gennewcandidate error candidate null\n");
            
            //printType(candidate[i]->progs[1]->root);
            //printprog(candidate[i]->progs[1]->root,0);
            //printf("before mutation\n");
            result[count] = mutation(candidate[i]);
            //if(!statementsTooLong(candidate[i]->root) && statementsTooLong(result[count]->root))
            //if(numOfWi(candidate[i]->root) == 1 && numOfWi(result[count]->root) != 1)
            if(existMultiCond(result[count]->root))
            {
                printf("mutype = %d\nbefore mutation:\n",mutype);
                printprog(candidate[i]->root,0,candidate[i]);
                printf("after mutation\n");
                printprog(result[count]->root,0,result[count]);
                exit(-1);
            }
            //printf("after mutation\n");
            /*if(existCS(candidate[i]->progs[1]->root) == 1 && existCS(result[count]->progs[1]->root) == 0)
             {
             printf("CS disappear\nbefore prog:\n");
             printType(candidate[i]->progs[1]->root);printf("\n");
             setFixed(candidate[i]->progs[1]->root);
             printType(candidate[i]->progs[1]->root);printf("\n");
             printprog(candidate[i]->progs[1]->root,0,candidate[i]->progs[1]);
             printf("after prog:\n");
             printprog(result[count]->progs[1]->root,0,result[count]->progs[1]);
             }*/
            
            //printf("xxx\n");
            setAll(result[count]);
            //printf("yyy\n");
            count++;
        }
    }
    free(selected);
    return result;
}


program** selectNewCandidate(int numofcandidate,program** candidate,int numofmutation, program** newcandidate)
{
    program** output = (program**)malloc(sizeof(program*) * numofcandidate);
    double* f = (double*)malloc(sizeof(double) * (numofcandidate + numofmutation));
    int *chosen1 = (int*)malloc(sizeof(int) * numofcandidate);
    int *chosen2 = (int*)malloc(sizeof(int) * numofmutation);
    int i,j;
    for(i = 0;i < numofcandidate;i++)
    {
        f[i] = candidate[i]->fitness;
        chosen1[i] = 0;
    }
    for(i = 0;i < numofmutation;i++)
    {
        f[i + numofcandidate] = newcandidate[i]->fitness;
        chosen2[i] = 0;
    }
    for(i = 0;i < numofcandidate + numofmutation - 1;i++)
        for(j = 0;j < numofcandidate + numofmutation - 1 - i;j++)
            if(f[j] < f[j + 1])
            {
                f[j] = f[j] + f[j + 1];
                f[j + 1] = f[j] - f[j + 1];
                f[j] = f[j] - f[j + 1];
            }
    double border = f[numofcandidate - 1];
    //for(i = 0;i < numofcandidate;i++) printf("f[%d]->f=%lf  ",i,candidate[i]->fitness);
    //for(i = 0;i < numofmutation;i++) printf("f[%d]->f=%lf  ",i,newcandidate[i]->fitness);
    printf("border=%lf\n",border);
    int count = 0;
    //printf("Choose new candidate:\n");
    for(i = 0;i < numofcandidate;i++)
        if(candidate[i]->fitness > border + 0.0001)
        {    //printf("candidate%d,",i);
            output[count] = candidate[i];
            count++;
            chosen1[i] = 1;
        }
    for(i = 0;i < numofmutation;i++)
        if(newcandidate[i]->fitness > border + 0.0001)
        {    //printf("new candidate%d,",i);
            output[count] = newcandidate[i];
            count++;
            chosen2[i] = 1;
        }
    
    for(i = 0;i < numofcandidate && count < numofcandidate;i++)
        if(candidate[i]->fitness > border - 0.0001 && candidate[i]->fitness < border + 0.0001)
        {    //printf("candidate%d,",i);
            output[count] = candidate[i];
            count++;
            chosen1[i] = 1;
        }
    for(i = 0;i < numofmutation && count < numofcandidate;i++)
        if(newcandidate[i]->fitness > border - 0.0001 && newcandidate[i]->fitness < border + 0.0001)
        {    //printf("new candidate%d,",i);
            output[count] = newcandidate[i];
            count++;
            chosen2[i] = 1;
        }
    //printf("ccount=%d",count);
    
    for(i = 0;i < numofcandidate;i++)
        if(chosen1[i] == 0)
            freeAll(NULL,candidate[i],NULL,NULL,NULL,2);
    for(i = 0;i < numofmutation;i++)
        if(chosen2[i] == 0)
            freeAll(NULL,newcandidate[i],NULL,NULL,NULL,2);
    free(f);
    free(chosen1);
    free(chosen2);
    return output;
}


program** genNewCandidateWithCoefficient1(int numofcandidate,program** candidate,int numofmutation,double coef)//no repeat
{
    int* selected = (int*)malloc(sizeof(int) * numofcandidate);
    program** result = (program**)malloc(sizeof(program*) * numofmutation);
    int i;
    for(i = 0;i < numofcandidate;i++)
        selected[i] = 0;
    
    int count = 0;
    while(count < numofmutation)
    {
        int index;
        do
        {
            index = nextrand(numofcandidate);
        }while(selected[index] == 1);
        
        selected[index] = 1;
        count++;
    }
    
    count = 0;
    for(i = 0;i < numofcandidate;i++)
    {
        if(selected[i] == 1)
        {
            
            if(candidate[i] == NULL)printf("gennewcandidate error candidate null\n");
            
            if(nextrand(100) < coef * 100)
                result[count] = mutation(candidate[i]);
            else
                result[count] = copyProgram(candidate[i]);
            
            setAll(result[count]);
            count++;
        }
    }
    free(selected);
    
    return result;
}

program** genNewCandidateWithCoefficient2(int numofcandidate,program** candidate,int numofmutation,double coef)//with repeat
{
    
    program** result = (program**)malloc(sizeof(program*) * numofmutation);
    int count = 0;
    
    while(count < numofmutation)
    {
        int index = nextrand(numofcandidate);
        
        if(nextrand(100) < coef * 100)
            result[count] = mutation(candidate[index]);
        else
            result[count] = copyProgram(candidate[index]);
        
        setAll(result[count]);
        count++;
    }
    
    return result;
}

program** selectNewCandidateWithFitness(int numofcandidate,program** candidate,int numofmutation, program** newcandidate)
{
    
    program** output = (program**)malloc(sizeof(program*) * numofcandidate);
    int* f = (int*)malloc(sizeof(double) * (numofcandidate + numofmutation));
    int *chosen = (int*)malloc(sizeof(int) * (numofcandidate + numofmutation));
    
    int addfitness = 0;
    int i,j;
    for(i = 0;i < numofcandidate;i++)
    {
        f[i] = candidate[i]->fitness + 8;
        chosen[i] = 0;
        addfitness += f[i];
    }
    for(i = 0;i < numofmutation;i++)
    {
        f[i + numofcandidate] = newcandidate[i]->fitness + 8;
        chosen[i + numofcandidate] = 0;
        addfitness += f[i + numofcandidate];
    }
    
    int* roulette = (int*)malloc(sizeof(int) * addfitness);
    int tempfit = 0;
    for(i = 0;i < numofcandidate;i++)
        for(j = 0;j < f[i];j++)
        {
            //printf("%dtempfit = %d,i=%d,j=%d\n",addfitness,tempfit,i,j);
            roulette[tempfit++] = i;
        }
    for(i = 0;i < numofmutation;i++)
    {
        for(j = 0;j < f[i + numofcandidate];j++)
        {
            //printf("%dtemfit = %d,i=%d,j=%d\n",addfitness,tempfit,i,j);
            roulette[tempfit++] = i + numofcandidate;
        }
    }
    
    if(tempfit != addfitness)
        printf("ERROR:TEMPFIT%d != ADDFITESS%d\n",tempfit,addfitness);
    
    int count = 0;
    while(count < numofcandidate)
    {
        int p = nextrand(addfitness);
        if(chosen[roulette[p]] == 0)
        {
            chosen[roulette[p]] = 1;
            count++;
        }
    }
    
    count = 0;
    for(i = 0;i < numofcandidate;i++)
    {
        if(chosen[i] == 1)
        {
            output[count] = candidate[i];
            count++;
        }
        else
            freeAll(NULL,candidate[i],NULL,NULL,NULL,2);
    }
    for(i = 0;i < numofmutation;i++)
    {
        if(chosen[i + numofcandidate] == 1)
        {
            output[count] = newcandidate[i];
            count++;
        }
        else
            freeAll(NULL,newcandidate[i],NULL,NULL,NULL,2);
    }
    
    free(f);
    free(chosen);
    free(roulette);
    
    return output;
}


//smc/Expr.c
Expr* createExpr(ExprType t,UnOp u,BinOp b,int bound,char* n,int v,Expr* child,Expr* left,Expr* right)
{
    Expr* result = (Expr*)malloc(sizeof(Expr));
    result->type = t;
    result->uop = u;
    result->bop = b;
    
    result->stepbound = bound;
    //strcpy(result->name,n);
    result->name = n;
    result->value = v;
    
    result->child = child;
    result->left = left;
    result->right = right;
    
    return result;
}
Expr* createExprConstant(int value){
    return createExpr(Constant,0,0,0,NULL,value,NULL,NULL,NULL);
}
Expr* createExprVar(char* name){
    return createExpr(Variable,0,0,0,name,0,NULL,NULL,NULL);
}
Expr* createExprExp1(UnOp op,Expr* child){
    return createExpr(Expr1,op,0,0,NULL,0,child,NULL,NULL);
}
Expr* createExprStepb(UnOp op,Expr* child,int stepbound){
    return createExpr(StepBoundExpr1,op,0,stepbound,NULL,0,child,NULL,NULL);
}
Expr* createExprExpr2(BinOp op,Expr* left,Expr* right){
    return createExpr(Expr2,0,op,0,NULL,0,NULL,left,right);
}

BinOp getBinOp(char* buf,int index)
{
    BinOp op;
    switch(buf[index])
    {
        case '-':if(buf[index + 1] == '>')
            op = Imp;
        else
            op = Min;
            break;
        case '&':op = And;break;
        case '|':op = Or;break;
        case '=':op = Eq;break;
        case '!':if(buf[index + 1] == '=')
        {
            op = Neq;
            break;
        }
        else
        {
            printf("Error:property BinOp !\n");
            exit(-1);
        }
        case '<':if(buf[index + 1] == '=')
            op = Le;
        else
            op = Lt;
            break;
        case '>':if(buf[index + 1] == '=')
            op = Ge;
        else
            op = Gt;
            break;
        case '+':op = Add;break;
        case '*':op = Mul;break;
        default:printf("Error:property BinOp\n");
            exit(-1);
    }
    return op;
}

int getBinOpLength(BinOp op)
{
    if(op == Neq || op == Le || op == Ge || op == Imp)
        return 2;
    else
        return 1;
}

Expr* generateStepExpr(char* buf, int leftindex, int rightindex)
{    //printf("%s,%d,%d\n",buf,leftindex,rightindex);
    if(buf[leftindex] == '(')
    {
        int index = leftindex,count = 0;
        while(index <= rightindex)
        {
            if(buf[index] == '(')
                count++;
            else if(buf[index] == ')')
                count--;
            if(count == 0)
                break;
            index++;
        }
        if(count != 0)
        {
            printf("Error:property parentheses not match!\n");
            exit(-1);
        }
        
        if(index == rightindex)
            return generateStepExpr(buf, leftindex + 1, rightindex - 1);
        else
        {
            Expr* left = generateStepExpr(buf, leftindex + 1, index - 1);
            BinOp op = getBinOp(buf,index + 1);
            Expr* right = generateStepExpr(buf, index + 1 + getBinOpLength(op),rightindex);
            return createExprExpr2(op,left,right);
        }
    }
    
    if(buf[leftindex] == 'G' || buf[leftindex] == 'F' || buf[leftindex] == 'X')
    {
        UnOp op;
        if(buf[leftindex] == 'G')
            op = Globally;
        else if(buf[leftindex] == 'F')
            op = Future;
        else
            op = Next;
        int index = leftindex + 4;
        int stepbound = 0;
        while(index <= rightindex && buf[index] <= '9' && buf[index] >= '0')
        {
            stepbound *= 10;
            stepbound += buf[index++] - '0';
        }
        
        Expr* child = generateStepExpr(buf,index,rightindex);
        Expr* result = createExprStepb(op,child,stepbound);
        
        return result;
    }
    else if(buf[leftindex] <= '9' && buf[leftindex] >= '0')
    {
        int value = 0;
        while(leftindex <= rightindex && buf[leftindex] <= '9' && buf[leftindex] >= '0')
        {
            value *= 10;
            value += buf[leftindex++] - '0';
        }
        return createExprConstant(value);
        
    }
    else if(buf[leftindex] <= 'z' && buf[leftindex] >= 'a')
    {
        int index = leftindex;
        while(index <= rightindex && (buf[index] <= '9' && buf[index] >= '0' || buf[index] <= 'z' && buf[index] >= 'a'))
            index++;
        char* name = (char*)malloc(sizeof(char) * (index - leftindex + 1));
        strncpy(name,&(buf[leftindex]),index - leftindex);
        name[index - leftindex] = '\0';
        Expr* left = createExprVar(name);
        if(index >= rightindex)
            return left;
        else
        {
            BinOp op = getBinOp(buf,index);
            Expr* right = generateStepExpr(buf,index + getBinOpLength(op),rightindex);
            return createExprExpr2(op,left,right);
        }
        /*while(index < rightindex)
         {
         BinOp op = getBinOp(buf,index);
         Expr* right = generateStepExpr(buf,index + getBinOpLength(op),rightindex);
         createExprExpr2(op,left,right);
         }*/
    }
    else
    {
        printf("Error:property!\n");
        exit(-1);
    }
}

int getStepBound(Expr* exp)
{
    if(exp == NULL)
        return 0;
    
    int left,right;
    switch(exp->type)
    {
        case Constant:return 0;
        case Variable:return 0;
        case Expr1:return getStepBound(exp->child);
        case StepBoundExpr1:return (exp->stepbound + getStepBound(exp->child));
        case Expr2:    left = getStepBound(exp->left);
            right = getStepBound(exp->right);
            return (left>right?left:right);
    }
}

void printUnOp(UnOp op)
{
    switch(op)
    {
        case Neg:printf("~");break;
        case Not:printf("!");break;
        case Future:printf("F");break;
        case Globally:printf("G");break;
        case Next:printf("X");break;
    }
}

void printBinOp(BinOp op)
{
    switch(op)
    {
        case Imp:printf("->");break;
        case And:printf("&");break;
        case Or:printf("|");break;
        case Eq:printf("=");break;
        case Neq:printf("!=");break;
        case Lt:printf("<");break;
        case Le:printf("<=");break;
        case Gt:printf(">");break;
        case Ge:printf(">=");break;
        case Add:printf("+");break;
        case Min:printf("-");break;
        case Mul:printf("*");break;
        case Div:printf("/");break;
    }
}
void printExpr(Expr* exp)
{
    if(exp == NULL)
        return;
    
    switch(exp->type)
    {
        case Constant:printf("%d",exp->value);break;
        case Variable:printf("%s",exp->name);break;
        case Expr1:printUnOp(exp->uop);printExpr(exp->child);break;
        case StepBoundExpr1:printf("(");printUnOp(exp->uop);printf("<=#%d",exp->stepbound);printExpr(exp->child);printf(")");break;
        case Expr2:printf("(");printExpr(exp->left);printBinOp(exp->bop);printExpr(exp->right);printf(")");break;
    }
}

int getExprValue(Expr* exp,trace* t,int currentStep)
{
    //printf("get expr value:");
    //printExpr(exp);
    //printf("\n");
    if(exp->type == Constant)
        return exp->value;
    else if(exp->type == Variable)
        return getVarvalueFromState(t->states[currentStep],exp->name);
    else if(exp->type == Expr1)
    {
        if(exp->uop == Neg)
        {
            return (-1 * getExprValue(exp->child,t,currentStep));
        }
        else if(exp->uop == Not)
        {
            if(getExprValue(exp->child,t,currentStep) == 0)
                return 1;
            else
                return 0;
        }
        else
        {
            printf("ERROR:Expr1 uop type invalid!\n");
            exit(-1);
        }
    }
    else if(exp->type == StepBoundExpr1)
    {
        
        if(exp->uop == Future)
        {
            int i;
            for(i = currentStep;i <= currentStep + exp->stepbound;i++)            //important
                if(getExprValue(exp->child,t,i) == 1)
                    return 1;
            return 0;
        }
        else if(exp->uop == Globally)
        {
            int i;
            for(i = currentStep;i <= currentStep + exp->stepbound;i++)            //important
                if(getExprValue(exp->child,t,i) == 0)
                    return 0;
            return 1;
        }
        else if(exp->uop == Next)
        {
            if(getExprValue(exp->child,t,currentStep + exp->stepbound) == 1)
                return 1;
            else
                return 0;
        }
        else
        {
            printf("ERROR:StepBoundExpr1 uop type invalid!\n");
            exit(-1);
        }
    }
    else if(exp->type == Expr2)
    {
        int left = getExprValue(exp->left,t,currentStep);
        int right;
        switch(exp->bop)
        {
            case Imp:    if(left > 0 && getExprValue(exp->right,t,currentStep) == 0)
                return  0;
            else
                return 1;
            case And:    if(left > 0 && getExprValue(exp->right,t,currentStep) > 0)
                return 1;
            else
                return 0;
            case Or:    if(left > 0 || getExprValue(exp->right,t,currentStep) > 0)
                return  1;
            else
                return 0;
            case Eq:    if(left == getExprValue(exp->right,t,currentStep))
                return  1;
            else
                return 0;
            case Neq:    if(left != getExprValue(exp->right,t,currentStep))
                return  1;
            else
                return 0;
            case Lt:    if(left < getExprValue(exp->right,t,currentStep))
                return  1;
            else
                return 0;
            case Le:    if(left <= getExprValue(exp->right,t,currentStep))
                return  1;
            else
                return 0;
            case Gt:    if(left > getExprValue(exp->right,t,currentStep))
                return  1;
            else
                return 0;
            case Ge:    if(left >= getExprValue(exp->right,t,currentStep))
                return  1;
            else
                return 0;
            case Add:    return (left + getExprValue(exp->right,t,currentStep));
            case Min:    return (left - getExprValue(exp->right,t,currentStep));
            case Mul:    return (left * getExprValue(exp->right,t,currentStep));
            case Div:    right = getExprValue(exp->right,t,currentStep);
                if(right != 0)
                    return (left / right);
                else
                    return 0;
            default:    printf("ERROR:getExprValue Expr2 BinOp type invalid!\n");
        }
    }
    else
    {
        printf("Error:Expr type invalid!\n");
        exit(-1);
    }
}
//smc/state.c
int getVarindexFromState(State* st,char* name)
{
    //printf("search %s in state\n",name);
    int i;
    for(i = 0;i < st->numvar;i++)
        if(strcmp(st->varname[i],name) == 0)
            break;
    if(i < st->numvar)
        return i;
    else
    {
        printf("ERROR variable name:%s search in state\n",name);
        for(i = 0;i < st->numvar;i++)
            printf("%d:%s\n",i,st->varname[i]);
        exit(-1);
    }
}

int getVarvalueFromState(State* st,char* name)
{
    //printf("xxx search %s in state\n",name);
    int i;
    for(i = 0;i < st->numvar;i++)
        if(strcmp(st->varname[i],name) == 0)
            break;
    if(i < st->numvar)
        return st->varvalue[i];
    else
    {
        printf("ERROR variable name:%s search in state\n",name);
        for(i = 0;i < st->numvar;i++)
            printf("%d:%s\n",i,st->varname[i]);
        exit(-1);
    }
}


//end

int fact(int n) {
    if (n <= 1) return 1;
    else return n*fact(n-1);
}

int my_mod(int x, int y) {
    return (x%y);
}

char *get_time()
{
    time_t ltime;
    time(&ltime);
    return ctime(&ltime);
}

double* set_coef(int numofrequirements)
{
    double* coef = (double*)malloc(sizeof(double) * numofrequirements);
    coef[0] = 0.6;coef[1] = 0.2;coef[2] = 0;coef[3] = 0;
    coef[4] = 0.2;coef[5] = -0.1;coef[6] = -0.2;
    // printf("x1\n");
    return coef;
}

Expr** set_requirments(int numofrequirements)
{
    int i,j;
    Expr** requirements = (Expr**)malloc(sizeof(Expr*) * numofrequirements);
    
    for(i = 0;i < numofrequirements;i++)
    {
        FILE* fp = NULL;
        char buf[255];
        switch(i)
        {
            case 0:fp = fopen("./property/M.bltl","r");break;
            case 1:fp = fopen("./property/B1.bltl","r");break;
            case 2:fp = fopen("./property/B2.bltl","r");break;
            case 3:fp = fopen("./property/B3.bltl","r");break;
            case 4:fp = fopen("./property/O1.bltl","r");break;
            case 5:fp = fopen("./property/O2.bltl","r");break;
            case 6:fp = fopen("./property/O3.bltl","r");break;
        }
        
        fgets(buf, 255, fp);
        fclose(fp);
        //printf("%s\n%d\n",buf,strlen(buf));
        int index = 0;
        j = 0;
        while(buf[j] != '\0')
        {
            if(buf[j] != ' ' && buf[j] != '\n')
            {
                buf[index] = buf[j];
                index++;
            }
            j++;
        }
        buf[index] = '\0';
        //printf("%s\n%d\n",buf,strlen(buf));
        
        Expr* e = generateStepExpr(buf,0,strlen(buf) - 1);
        //printf("xy,%d,%d\n",i,sizeof(requirements));
        requirements[i] = e;
        //printExpr(e);
        //printf("\n");
        
    }
    
    int maxstepbound = 0;
    for(i = 0;i < numofrequirements;i++)
    {
        int step = getStepBound(requirements[i]);
        if(step > maxstepbound)
            maxstepbound = step;
    }
    //printf("max step bound:%d\n",maxstepbound);
    initTraceGlobalVar(maxstepbound + 5);
    
    
    return requirements;
}


int mutationCond_(cond* root,program* prog,int type, int action)    //type == 1:can add        type == 0:can't add
{
    long condRandom = 0;
    int illegal = 0;
    if(root == NULL)
        return 1;
    if(root->type == -1)
        return 1;
    if(root->type == 0)
    {    //printf("0");
        cond* new = gencond(prog,3);
        while(new->type == 0)
        {
            freeAll(NULL, NULL, NULL, new, NULL, 4);
            // free(new);
            new = gencond(prog,3);
        }
        root->type = new->type;
        root->exp1 = new->exp1;
        root->exp2 = new->exp2;
        root->cond1 = new->cond1;
        root->cond2 = new->cond2;
        // freeAll(NULL, NULL, NULL, new, NULL, 4);
        free(new);
    }
    if (action > 2 && action < 12)
    {
        if (root->type == 1 || root->type == 2)
            switch (action)
        {
            case 3:
                root->exp1 = genexp_(prog, 1, 10);
                break;
            case 4:
                root->exp1 = genexp_(prog, 1, 11);
                break;
            case 5:
                root->exp1 = genexp_(prog, 1, 12);
                break;
            case 6:
                root->exp1 = genexp_(prog, 1, 13);
                break;
            case 7:
                root->exp1 = genexp_(prog, 1, 14);
                break;
            case 8:
                root->exp2 = genexp_(prog, 0, 0);
                break;
            case 9:
                root->exp2 = genexp_(prog, 0, 1);
                break;
            case 10:
                root->exp2 = genexp_(prog, 0, 2);
                break;
            case 11:
                root->exp2 = genexp_(prog, 0, 3);
                break;
        }
        else if (root->type == 3 || root->type == 4)
        {
            switch (action)
            {
                case 3:
                    root->cond1->exp1 = genexp_(prog, 1, 10);
                    break;
                case 4:
                    root->cond1->exp1 = genexp_(prog, 1, 11);
                    break;
                case 5:
                    root->cond1->exp1 = genexp_(prog, 1, 12);
                    break;
                case 6:
                    root->cond1->exp1 = genexp_(prog, 1, 13);
                    break;
                case 7:
                    root->cond1->exp1 = genexp_(prog, 1, 14);
                    break;
                case 8:
                    root->cond1->exp2 = genexp_(prog, 0, 0);
                    break;
                case 9:
                    root->cond1->exp2 = genexp_(prog, 0, 1);
                    break;
                case 10:
                    root->cond1->exp2 = genexp_(prog, 0, 2);
                    break;
                case 11:
                    root->cond1->exp2 = genexp_(prog, 0, 3);
                    break;
            }
        }
        else
            illegal = 1;
    }
    else if (action > 11 && action < 21)
    {
        if (root->type == 3 || root->type == 4)
        {
            
            switch (action)
            {
                case 12:
                    
                    root->cond2->exp1 = genexp_(prog, 1, 10);
                    break;
                case 13:
                    root->cond2->exp1 = genexp_(prog, 1, 11);
                    break;
                case 14:
                    root->cond2->exp1 = genexp_(prog, 1, 12);
                    break;
                case 15:
                    root->cond2->exp1 = genexp_(prog, 1, 13);
                    break;
                case 16:
                    root->cond2->exp1 = genexp_(prog, 1, 14);
                    break;
                case 17:
                    root->cond2->exp2 = genexp_(prog, 0, 0);
                    break;
                case 18:
                    root->cond2->exp2 = genexp_(prog, 0, 1);
                    break;
                case 19:
                    root->cond2->exp2 = genexp_(prog, 0, 2);
                    break;
                case 20:
                    
                    root->cond2->exp2 = genexp_(prog, 0, 3);
                    break;
            }
        }
        else
            illegal = 1;
    }
    else if (action > 20 && action < 24)
    {
        if (root->type == 1 || root->type == 2)
        {
            cond* c1 = NULL;
            switch (action)
            {
                case 21:
                    root->type = 3 - root->type;
                    break;
                case 22:
                    // freeAll(NULL, NULL, NULL, root, NULL, 4);
                    // free(root);
                    free(root->exp1);
                    free(root->exp2);
                    cond* new;
                    if(type == 1)
                    {
                        condRandom = 321112100;
                        new = gencond_(prog,5, condRandom);
                    }
                    else
                    {
                        condRandom = 2100;
                        new = gencond_(prog,3, condRandom );
                    }
                    root->type = new->type;
                    root->exp1 = new->exp1;
                    root->exp2 = new->exp2;
                    root->cond1 = new->cond1;
                    root->cond2 = new->cond2;
                    //freeAll(NULL, NULL, NULL, new, NULL, 4);
                    free(new);
                    break;
                case 23:
                    condRandom = 2100;
                    c1 = copyCond(root);
                    free(root->exp1);
                    root->exp1 = NULL;
                    free(root->exp2);
                    root->exp2 = NULL;
                    // root->type = nextrand(2) + 3;
                    root->type = 3;
                    root->cond1 = c1;
                    root->cond2 = gencond_(prog,3, condRandom);
                    break;
                case 24:
                    condRandom = 2100;
                    c1 = copyCond(root);
                    // freeAll(NULL, NULL, NULL, NULL, root->exp1, 5);
                    free(root->exp1);
                    root->exp1 = NULL;
                    freeAll(NULL, NULL, NULL, NULL, root->exp2, 5);
                    // free(root->exp2);
                    root->exp2 = NULL;
                    // root->type = nextrand(2) + 3;
                    root->type = 4;
                    root->cond1 = c1;
                    root->cond2 = gencond_(prog,3, condRandom);
                    break;
            }
        }
        else
            illegal = 1;
    }
    else if (action > 24 && action < 31)
    {
        if (root->type == 3 || root->type == 4)
        {
            switch (action)
            {
                case 25:
                    mutationCond_(root->cond1,prog,0, 22);
                    break;
                case 26:
                    root->type = 7 - root->type;
                    break;
                case 27:
                    mutationCond_(root->cond2,prog,0, 22);
                    break;
                case 28:
                    root->type = root->cond1->type;
                    root->exp1 = root->cond1->exp1;
                    root->exp2 = root->cond1->exp2;
                    // freeAll(NULL, NULL, NULL, root->cond1, NULL, 4);
                    free(root->cond1);
                    root->cond1 = NULL;
                    freeAll(NULL, NULL, NULL, root->cond2, NULL, 4);
                    root->cond2 = NULL;
                    break;
                case 29:
                    root->type = root->cond2->type;
                    root->exp1 = root->cond2->exp1;
                    root->exp2 = root->cond2->exp2;
                    freeAll(NULL, NULL, NULL, root->cond1, NULL, 4);
                    root->cond1 = NULL;
                    // freeAll(NULL, NULL, NULL, root->cond2, NULL, 4);
                    free(root->cond2);
                    root->cond2 = NULL;
                    break;
                case 30:
                    // free(root->cond1);
                    // free(root->cond2);
                    freeAll(NULL, NULL, NULL, root->cond1, NULL, 4);
                    freeAll(NULL, NULL, NULL, root->cond2, NULL, 4);
                    condRandom = 311212100;
                    cond* new = gencond_(prog,5, condRandom);
                    root->type = new->type;
                    root->exp1 = new->exp1;
                    root->exp2 = new->exp2;
                    root->cond1 = new->cond1;
                    root->cond2 = new->cond2;
                    // freeAll(NULL, NULL, NULL, new, NULL, 4);
                    free(new);
                    break;
            }
        }
        else
            illegal = 1;
    }
    return  illegal;
    
}
//mutation 1
program* mutation1(program* parent, int nodeNum, int actionNum)
{
    int mutationtype = 0;
    int illegal = 0;
    int commandtypeVarindex[2];
    int assignRandom;
    long condrandom;
    program* newprog = copyProgram(parent);
    newprog->checkedBySpin = 0;
    treenode* new = newprog->root;
    treenode* chnode = findNode(newprog->root, newprog, nodeNum);
    if (chnode == NULL) {
        newprog->illegal = 1;
        return newprog;
    }
    mutype = mutationtype;
    newprog->illegal = 0;
    //Replacement Mutation type
    treenode* mnode = chnode;
    if (actionNum >= 0 && actionNum < 3 )
    {
        if(mnode->fixed == 0)
        {
            treenode* newnode = NULL;
            switch (actionNum)
            {
                case 0:
                    commandtypeVarindex[0] = 2;
                    assignRandom = 0;
                    newnode = genprog_(mnode->depth,newprog, commandtypeVarindex, assignRandom,NULL, 0);
                    break;
                case 1:
                    if (mnode->depth == 3) {
                        commandtypeVarindex[0] = 10;
                        commandtypeVarindex[1] = 2;
                        assignRandom = 11;
                        condrandom = 311001111;
                        newnode = genprog_(newprog->maxdepth,newprog, commandtypeVarindex, assignRandom,condrandom, 0);
                    }
                    else
                    {
                        newprog->illegal = 1;
                        return newprog;
                    }
                    break;
                case 2:
                    if (mnode->depth == 3) {
                        commandtypeVarindex[0] = 20;
                        commandtypeVarindex[1] = 2;
                        assignRandom = 11;
                        condrandom = 311001111;
                        newnode = genprog_(newprog->maxdepth,newprog, commandtypeVarindex, assignRandom,condrandom, 0);
                    }
                    else
                    {
                        newprog->illegal = 1;
                        return newprog;
                    }
                    break;
            }
            if(mnode->parent == NULL)
                new = newnode;
            else
            {
                if(mnode->parent->treenode1 == mnode)
                    mnode->parent->treenode1 = newnode;
                else
                    mnode->parent->treenode2 = newnode;
                newnode->parent = mnode->parent;
                free(mnode);
            }
        }
        else
        {
            newprog->illegal = 1;
            return newprog;
        }
    }
    else if (actionNum >= 3 && actionNum < 31 )
    {
        if(mnode->cond1 != NULL)
        {
            newprog->illegal = mutationCond_(mnode->cond1,newprog,1,actionNum);
            if (newprog->illegal == 1)
            {
                return newprog;
            }
        }
        else
        {
            newprog->illegal = 1;
            return newprog;
        }
    }
    else if (actionNum >= 31 && actionNum < 40)
    {
        if (mnode->exp1 != NULL)
        {
            exp_* e = NULL;
            switch (actionNum)
            {
                case 31:
                    e = genexp_(newprog,0,0);
                    break;
                case 32:
                    e = genexp_(newprog,0,1);
                    break;
                case 33:
                    e = genexp_(newprog,0,2);
                    break;
                case 34:
                    e = genexp_(newprog,0,3);
                    break;
                case 35:
                    mnode->index = 0;
                    break;
                case 36:
                    mnode->index = 1;
                    break;
                case 37:
                    mnode->index = 2;
                    break;
                case 38:
                    mnode->index = 3;
                    break;
                case 39:
                    mnode->index = 4;
                    break;
            }
            if (actionNum >= 31 && actionNum < 35) {
                freeAll(NULL, NULL, NULL, NULL, mnode->exp1, 5);
                mnode->exp1 = e;
            }
        }
        else
        {
            newprog->illegal = 1;
            return newprog;
        }
    }
    //Insert Mutation types
    treenode* newnode;
    if (actionNum > 39 && actionNum < 42)
    {
        if(mnode->fixed == 1 || mnode->depth + mnode->height == newprog->maxdepth + 1 || mnode->depth == 2)
        {
            newprog->illegal = 1;
            return newprog;
        }
        else
        {
            switch (actionNum)
            {
                case 40:
                    newnode = createTreenode(0,0,NULL,NULL,NULL,NULL);
                    condrandom = 311001111;
                    newnode->cond1 = gencond_(newprog,5,condrandom);
                    newnode->treenode1 = mnode;
                    if(mnode->parent == NULL)
                    {    //printf("first\n");
                        mnode->parent = newnode;
                        new = newnode;
                    }
                    else
                    {
                        if(mnode->parent->treenode1 == mnode)
                            mnode->parent->treenode1 = newnode;
                        else if(mnode->parent->treenode2 == mnode)
                            mnode->parent->treenode2 = newnode;
                        else
                            printf("mutation type2 error!\n");
                        
                        newnode->parent = mnode->parent;
                        mnode->parent = newnode;
                        
                    }
                    break;
                    
                case 41:
                    newnode = createTreenode(1,0,NULL,NULL,NULL,NULL);
                    condrandom = 311001111;
                    newnode->cond1 = gencond_(newprog,5,condrandom);
                    newnode->treenode1 = mnode;
                    if(mnode->parent == NULL)
                    {    //printf("first\n");
                        mnode->parent = newnode;
                        new = newnode;
                    }
                    else
                    {
                        if(mnode->parent->treenode1 == mnode)
                            mnode->parent->treenode1 = newnode;
                        else if(mnode->parent->treenode2 == mnode)
                            mnode->parent->treenode2 = newnode;
                        else
                            printf("mutation type2 error!\n");
                        
                        newnode->parent = mnode->parent;
                        mnode->parent = newnode;
                        
                    }
                    break;
            }
            
        }
        
    }
    else if(actionNum > 41 && actionNum < 48 )
    {
        if(!(mnode->depth != 1 && mnode->numofstatements < 6 && mnode->depth == 2) && !(mnode->depth != 1 && mnode->numofstatements < 2 && mnode->depth != 2))
        {
            illegal = 1;
            return newprog;
        }
        else{
            switch (actionNum)
            {
                case 42:
                    newnode = createTreenode(3,0,NULL,NULL,NULL,NULL);
                    if(getStatement(mnode,0)->type != 1)
                    {
                        newnode->treenode1 = mnode;
                        commandtypeVarindex[0] = 2;
                        assignRandom = 0;
                        newnode->treenode2 = genprog_(mnode->depth,newprog, commandtypeVarindex, assignRandom, NULL, 0);
                    }
                    else
                    {
                        newprog->illegal = 1;
                        return newprog;
                    }
                    break;
                case 43:
                    newnode = createTreenode(3,0,NULL,NULL,NULL,NULL);
                    if(getStatement(mnode,0)->type != 1)
                    {
                        if (mnode->depth == 3)
                        {
                            newnode->treenode1 = mnode;
                            commandtypeVarindex[0] = 10;
                            commandtypeVarindex[1] = 2;
                            assignRandom = 11;
                            condrandom = 311001111;
                            newnode->treenode2 = genprog_(newprog->maxdepth,newprog, commandtypeVarindex, assignRandom,condrandom, 0);
                        }
                        else
                        {
                            newprog->illegal = 1;
                            return newprog;
                        }
                    }
                    else
                    {
                        newprog->illegal = 1;
                        return newprog;
                    }
                    break;
                case 44:
                    newnode = createTreenode(3,0,NULL,NULL,NULL,NULL);
                    if(getStatement(mnode,0)->type != 1)
                    {
                        if (mnode->depth == 3)
                        {
                            newnode->treenode1 = mnode;
                            commandtypeVarindex[0] = 20;
                            commandtypeVarindex[1] = 2;
                            assignRandom = 11;
                            condrandom = 311001111;
                            newnode->treenode2 = genprog_(newprog->maxdepth,newprog, commandtypeVarindex, assignRandom,condrandom, 0);
                        }
                        else
                        {
                            newprog->illegal = 1;
                            return newprog;
                        }
                    }
                    else
                    {
                        newprog->illegal = 1;
                        return newprog;
                    }
                    break;
                case 45:
                    newnode = createTreenode(3,0,NULL,NULL,NULL,NULL);
                    if(getStatement(mnode,0)->type != 5)
                    {
                        newnode->treenode2 = mnode;
                        commandtypeVarindex[0] = 2;
                        assignRandom = 0;
                        newnode->treenode1 = genprog_(mnode->depth,newprog, commandtypeVarindex, assignRandom, NULL, 0);
                    }
                    else
                    {
                        newprog->illegal = 1;
                        return newprog;
                    }
                    break;
                case 46:
                    newnode = createTreenode(3,0,NULL,NULL,NULL,NULL);
                    if(getStatement(mnode,0)->type != 5)
                    {
                        if (mnode->depth == 3)
                        {
                            newnode->treenode2 = mnode;
                            commandtypeVarindex[0] = 10;
                            commandtypeVarindex[1] = 2;
                            assignRandom = 11;
                            condrandom = 311001111;
                            newnode->treenode1 = genprog_(newprog->maxdepth,newprog, commandtypeVarindex, assignRandom,condrandom, 0);
                        }
                        else
                        {
                            newprog->illegal = 1;
                            return newprog;
                        }
                    }
                    else
                    {
                        newprog->illegal = 1;
                        return newprog;
                    }
                    break;
                case 47:
                    newnode = createTreenode(3,0,NULL,NULL,NULL,NULL);
                    if(getStatement(mnode,0)->type != 5)
                    {
                        if (mnode->depth == 3)
                        {
                            newnode->treenode2 = mnode;
                            commandtypeVarindex[0] = 20;
                            commandtypeVarindex[1] = 2;
                            assignRandom = 11;
                            condrandom = 311001111;
                            newnode->treenode1 = genprog_(newprog->maxdepth,newprog, commandtypeVarindex, assignRandom,condrandom, 0);
                        }
                        else
                        {
                            newprog->illegal = 1;
                            return newprog;
                        }
                    }
                    else
                    {
                        newprog->illegal = 1;
                        return newprog;
                    }
                    break;
            }
        }
        if(mnode->parent == NULL)
        {
            mnode->parent = newnode;
            new = newnode;
        }
        else
        {
            if(mnode->parent->treenode1 == mnode)
                mnode->parent->treenode1 = newnode;
            else
                mnode->parent->treenode2 = newnode;
            newnode->parent = mnode->parent;
            mnode->parent = newnode;
        }
    }
    //Reduction Mutation type
    else if(actionNum > 47 && actionNum < 50 )
    {
        
        switch (actionNum) {
            case 48:
                if (mnode->treenode1 != NULL && mnode->treenode1->fixed == 0 && (mnode->treenode2 == NULL || mnode->treenode2->fixed == 0))
                {
                    if(mnode->parent == NULL)    //mnode == new
                    {
                        new = mnode->treenode1;
                        new->parent = NULL;
                    }
                    else                         //mnode != new
                    {
                        mnode->treenode1->parent = mnode->parent;
                        if(mnode->parent->treenode1 == mnode)
                            mnode->parent->treenode1 = mnode->treenode1;
                        else
                            mnode->parent->treenode2 = mnode->treenode1;
                    }
                    free(mnode);
                }
                else
                {
                    newprog->illegal = 1;
                    return newprog;
                }
                break;
            case 49:
                if (mnode->treenode2 != NULL && mnode->treenode2->fixed == 0 && (mnode->treenode1 == NULL && mnode->treenode1->fixed == 0 ))
                {
                    if(mnode->parent == NULL)    //mnode == new
                    {
                        new = mnode->treenode2;
                        new->parent = NULL;
                    }
                    else                         //mnode != new
                    {
                        mnode->treenode2->parent = mnode->parent;
                        if(mnode->parent->treenode1 == mnode)
                            mnode->parent->treenode1 = mnode->treenode2;
                        else
                            mnode->parent->treenode2 = mnode->treenode2;
                    }
                    // freeAll(NULL, NULL, mnode, NULL, NULL, 3);
                    free(mnode);
                }
                else
                {
                    newprog->illegal = 1;
                    return newprog;
                }
                
                break;
        }
    }
    newprog->root = new;
    return newprog;
}

long setConrandom(int a, int b, int c, int d, int e, int f, int g)
{
    char* str = (char*)malloc(sizeof(char)*9);
    sprintf(str, "%d%d%d%d%d%d%d", a, b, c, d, e, f, g);
    return strtoul(str,NULL,10);
}

action* setAction(int layer, int actionType)
{
    long num = 3200*24;
    action* result;
    action* nodeAction = (action*)malloc(sizeof(action));
    // action** nodeAction24 = (action**)malloc(sizeof(action*)*6);
    // action** nodeAction3 = (action**)malloc(sizeof(action*)*num);
    int commandtypeVarindex_[4] = { 2, 3, 10, 20};
    int commandtypeVarindex24[2] = {2, 3};
    int assignRandom_[3] = { 0, 1, 2};
    long condrandom_[3200];
    int a[2] = {3, 4};
    int b[2] = {1, 2};
    int c[5] = {10, 11, 12, 13, 14};
    int d[4] = {0, 1, 2, 3};
    int k = 0;
    while (k < 3200) {
        for(int i=0; i<2; i++)
        {
            for (int j=0 ; j<2; j++)
            {
                for (int n=0; n< 5; n++)
                {
                    for (int m=0; m<4; m++)
                    {
                        for (int q=0 ; q<2; q++)
                        {
                            for (int w=0; w< 5; w++)
                            {
                                for (int e=0; e<4; e++)
                                {
                                    condrandom_[k]=setConrandom(a[i], b[j], c[n], d[m], b[q], c[w], d[e]);
                                    k++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    /*// layer 2 4 actions
     int m = 0;
     for(int i=0; i<2; i++)
     {
     for(int j=0; j<3; j++)
     {
     //  nodeAction24[m] = (action*)malloc(sizeof(action));
     //  nodeAction24[m]->commandtypeVarindex[0] = commandtypeVarindex24[i];
     //  nodeAction24[m]->assignRandom = assignRandom_[j];
     m++;
     }
     }*/
    if (layer == 2 || layer == 4) {
        nodeAction->commandtypeVarindex[0] = commandtypeVarindex24[actionType / 3];
        nodeAction->assignRandom = assignRandom_[actionType % 3];
    }
    else{
        nodeAction->condrandom = condrandom_[actionType % 3200];
        nodeAction->assignRandom = assignRandom_[actionType / 3200 % 3];
        nodeAction->commandtypeVarindex[1] = commandtypeVarindex24[actionType / 3200 / 3 % 2];
        nodeAction->commandtypeVarindex[0] = commandtypeVarindex_[actionType / 3200 / 3 / 2 % 4];
    }
    // layer 3 actions
    /*
     m = 0;
     for (int i=0; i< 4; i++)
     {
     for (int j=0; j<2; j++)
     {
     for (int r=0; r<3; r++)
     {
     for (int c=0; c<3200; c++)
     {
     nodeAction3[m] =(action*)malloc(sizeof(action));
     nodeAction3[m]->commandtypeVarindex[0] = commandtypeVarindex_[i];
     nodeAction3[m]->commandtypeVarindex[1] = commandtypeVarindex24[j];
     nodeAction3[m]->assignRandom = assignRandom_[r];
     nodeAction3[m]->condrandom = condrandom_[c];
     m++;
     }
     }
     }
     }
     */
    
    result = nodeAction;
    
    return result;
}

program* initProg(Expr** requirements ,int numofrequirements,double* coef)
{
    int numofcandidate = 1;
    action* act = (action*)malloc(sizeof(action));
    program** candidate = genInitTemplate(numofcandidate);
    for(int i = 0;i < numofcandidate;i++)
    {
        
        organism* org = genOrganism(candidate[i]);
        double candidatefit = calculateFitness(org,requirements,numofrequirements,coef);
        candidate[i]->fitness = candidatefit;
        //printf("init candidate %d:%lf\n",i,candidate[i]->fitness);
        for(int j = 0;j < numofrequirements;j++)
            candidate[i]->propertyfit[j] = org->progs[0]->propertyfit[j];
        candidate[i]->checkedBySpin = 0;
        freeAll(org,NULL,NULL,NULL,NULL,1);
        
    }
    printAst(candidate[0]);
    return candidate[0];
}

program* mutation_(program* candidate0, int nodeNum, int actType, Expr** requirements ,int numofrequirements,double* coef)
{
    program* newcandidate = NULL;
    newcandidate = mutation1(candidate0, nodeNum, actType);
    setAll(newcandidate);
    printAst(newcandidate);
    organism* org = genOrganism(newcandidate);
    double candidatefit1 = calculateFitness(org,requirements,numofrequirements,coef);
    newcandidate->fitness = candidatefit1;
    for(int j = 0;j < numofrequirements;j++)
        newcandidate->propertyfit[j] = org->progs[0]->propertyfit[j];
    freeAll(NULL, candidate0, NULL, NULL, NULL, 2);
    freeAll(org,NULL,NULL,NULL,NULL,1);
    return newcandidate;
}

int spin_(program* candidate)
{
    int numofspin = 0;
    int numofsolution = 0;
    int right = 0;
    int reward = 0;
    
    if(candidate->propertyfit[0] >0.99999 && candidate->checkedBySpin == 0 && candidate->fitness > 78.4)
    {
        candidate->checkedBySpin = 1;
        organism* org = genOrganism(candidate);
        
        /*printf("%d,Template:\n",i);
         printprog(candidate[j]->root,0,candidate[j]);
         printf("program0:\n");
         printprog(org->progs[0]->root,0,org->progs[0]);
         printf("program1:\n");
         printprog(org->progs[1]->root,0,org->progs[1]);*/
        
        // FILE* f;
        char filename[60] = "./output/mutex.pml";
        FILE *f;
        // f = fopen("/Users/zhuang/workspace-gp/testSwig2/output/mutex.pml","w+");
        if(f = fopen( filename, "w"))
        {
            orgToPml(org,f);
        }
        //sleep(2);
        fclose(f);
        
        char command[100] = "spin -a ";
        strcat(command,filename);
        strcat(command," > useless");
        
        
        system(command);
        system("gcc -DMEMLIM=1024 -O2 -DXUSAFE -w -o pan pan.c");
        
        numofspin++;
        system("./pan -m10000 -a -f -N e1 > pan1.out");
        int r1 = system("grep -q -e \"errors: 0\" pan1.out");
        if(r1 == 0)
            reward += 10;
        
        numofspin++;
        system("./pan -m10000 -a -f -N e2 > pan2.out");
        int r2 = system("grep -q -e \"errors: 0\" pan2.out");
        if(r2 == 0)
            reward += 5;
        
        numofspin++;
        system("./pan -m10000 -a -f -N e3 > pan3.out");
        int r3 = system("grep -q -e \"errors: 0\" pan3.out");
        if(r3 == 0)
            reward += 5;
        
        if (reward == 20) {
            printprog(candidate->root,0,candidate);
            printprog(org->progs[0]->root,0,org->progs[0]);
            printprog(org->progs[1]->root,0,org->progs[1]);
            printf("\n");
            right = 1;
        }
        
        if (right == 1) {
            FILE* f;
            char filename[100] = "./output/mutexCorrect";
            // strcat(filename,iToStr(1));
            strcat(filename,".pml");
            if(f = fopen(filename,"w"))
            {
                orgToPml(org,f);
            }
            fclose(f);
        }
        
        freeAll(org,NULL,NULL,NULL,NULL,1);
    }
    
    return reward;
}
