#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <vector>

#include "Eigen/Dense"
#include "Eigen/Sparse"

#include "ftrl.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
typedef Eigen::Triplet<double> T;

float alpha = 0.05;  // learning rate
float beta = 1.0;    // smoothing parameter for adaptive learning rate
float L1 = 0.5;      // L1 regularization, larger value means more regularized
float L2 = 0.3;      // L2 regularization, larger value means more regularized

long D = 2000*10000; // number of weights to use

long holdout = 100;  // use every N training instance for holdout validation

static char *line = NULL;
static int max_line_len;
static char* readline(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

int main(int argc, char* argv[]) {
    const char *filename = "./data/converted";
    FILE *fp = fopen(filename,"r");
    max_line_len = 1024;
    line = Malloc(char,max_line_len);

    ftrl_proximal learner;
    learner.init(alpha,beta,L1,L2,D);

    char *idx, *val,*endptr;
    double loss = 0.0;
    long count = 0, t = 0;
	while(readline(fp)!=NULL)
	{
        char *p = strtok(line," \t\n"); // label
        float y = strtof(p,&endptr);

        SparseVector<float> x(D);
		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

            if(val == NULL)
                break;

            long index = strtol(idx,&endptr,10);
            int value = strtod(val,&endptr);
            x.coeffRef(index) = value;
		}

        float pro = learner.predict(x);
        if (t % holdout == 0) {
           loss += learner.logloss(pro,y); 
           count += 1;
        } else  {
            learner.update(x, pro, y);
        }
        if (t % (holdout*10) == 0 and t > 1) {
            printf("current:%d logloss: %f\n", t, loss/count);
        }
        t++;
	}

    return 0;
}
