// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "corpus.h"

using namespace std;

int topic = 100;

vector<vector<int> > nkt;
vector<int> nk;

vector<vector<int> > z;
vector<int> zk;

void initializeModel(corpus &co) {
    int num_terms = co.num_terms;

    nk.resize(topic);
    nkt.resize(topic);
    for (int k = 0; k < topic; ++k) {
        nkt[k].resize(num_terms);
    }

    zk.resize(topic);
    int doc_num = co.docs.size();
    z.resize(doc_num);
    for (int d = 0; d < doc_num; ++d) {
        int N = co.docs[d].size();
        z[d].resize(N);
        for (int n = 0; n < N; ++n) {
            int initTopic = (static_cast<double>(random()) / RAND_MAX) * topic;
            z[d][n] = initTopic;
            zk[initTopic] ++;
            nkt[initTopic][co.biterms[co.docs[d][n]].first] ++;
            nkt[initTopic][co.biterms[co.docs[d][n]].second] ++;
            nk[initTopic] += 2;
        }
    }
}

void inferenceModel(corpus &co,double alpha, double beta) {
    double *prob = new double[topic];

    for (size_t d = 0; d < co.docs.size(); ++d) {
        for (size_t n = 0; n < co.docs[d].size(); ++n) {
            int t1 = co.biterms[co.docs[d][n]].first;
            int t2 = co.biterms[co.docs[d][n]].second;
            int oldTopic = z[d][n];
            int V = co.num_terms;

            zk[oldTopic] --;
            nkt[oldTopic][t1] --;
            nkt[oldTopic][t2] --;
            nk[oldTopic] -= 2 ;

            for (int k = 0; k < topic; k ++){
                prob[k] = (nkt[k][t1] + beta) / (nk[k] + V * beta) * (nkt[k][t2] + beta) / (nk[k] + V * beta)  * (zk[k] + alpha);
            }

            for (int k = 1; k < topic; k ++){
                prob[k] += prob[k - 1];
            }
            
            double u = (static_cast<double>(random()) / RAND_MAX) * prob[topic - 1];
            int newTopic;
            for (newTopic = 0; newTopic < topic; newTopic++){
                if (u < prob[newTopic])
                    break;
            }

            zk[newTopic] ++;
            nkt[newTopic][t1] ++;
            nkt[newTopic][t2] ++;
            nk[newTopic] += 2 ;
            z[d][n] = newTopic;
        }
    }

    delete []prob;
}

int main(int argc, char* argv[]) {
    const char *biterm = "/data0/data/btm/bitermIndex";
    const char *doc = "/data0/data/btm/corpus";
    int iter = 100;
    double alpha = 0.01;
    double beta = 0.05;

    struct corpus co;
    load_corpus(biterm,doc,co);
    initializeModel(co);

    for (int i = 0; i < iter; ++i) {
        printf("%d\n",i);
        inferenceModel(co,alpha,beta);
    }

    return 1; 
}
