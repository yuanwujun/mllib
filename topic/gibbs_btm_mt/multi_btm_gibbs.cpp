// Copyright 2014 yuanwujun. All Rights Reserved.
// Author: real.yuanwj@gmail.com (WuJun Yuan)

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <omp.h>

#include "corpus.h"

using namespace std;

int topic = 300;
double alpha = 0.01;
double beta = 0.05;

vector<vector<int> > z;

struct processor{
    vector<vector<int> > nkt;
    vector<int> nk;
    vector<int> zk;
};

void collect_nkt (corpus &co, processor &td) {
    int V = co.num_terms;
    td.nk.resize(topic);
    td.nkt.resize(topic);
    td.zk.resize(topic);
    for (int k = 0; k < topic; ++k) {
        td.nk[k] = 0;
        td.zk[k] = 0;
        td.nkt[k].resize(V);
        for (int v = 0; v < V; ++v) {
            td.nkt[k][v] = 0;
        }
    }

    size_t doc_num = z.size();
    for (size_t d = 0; d < doc_num; ++d) {
        int N = co.docs[d].size();
        for (int n = 0; n < N; ++n) {
            int newTopic = z[d][n];
            int t1 = co.biterms[co.docs[d][n]].first;
            int t2 = co.biterms[co.docs[d][n]].second;

            td.zk[newTopic] ++;
            td.nkt[newTopic][t1] ++;
            td.nkt[newTopic][t2] ++;
            td.nk[newTopic] += 2;
        }
    }
}

void initializeModel(corpus &co) {
    int doc_num = co.docs.size();
    z.resize(doc_num);

    #pragma omp parallel for
    for (int d = 0; d < doc_num; ++d) {
        int N = co.docs[d].size();
        z[d].resize(N);
        for (int n = 0; n < N; ++n) {
            z[d][n] = (static_cast<double>(random()) / RAND_MAX) * topic;
        }
    }
}

void inferenceModel(corpus &co, int d, processor &td) {
    double *prob = new double[topic];

    for (size_t n = 0; n < co.docs[d].size(); ++n) {
        int t1 = co.biterms[co.docs[d][n]].first;
        int t2 = co.biterms[co.docs[d][n]].second;
        int oldTopic = z[d][n];
        int V = co.num_terms;

        td.zk[oldTopic] --;
        td.nkt[oldTopic][t1] --;
        td.nkt[oldTopic][t2] --;
        td.nk[oldTopic] -= 2 ;

        for (int k = 0; k < topic; k ++){
            prob[k] = (td.nkt[k][t1] + beta) / (td.nk[k] + V * beta) * 
                (td.nkt[k][t2] + beta) / (td.nk[k] + V * beta)  * (td.zk[k] + alpha);
        }

        for (int k = 1; k < topic; k ++){
            prob[k] += prob[k - 1];
        }
            
        double u = (static_cast<double>(random()) / RAND_MAX) * prob[topic - 1];
        int newTopic;
        for (newTopic = 0; newTopic < topic; newTopic++){
            if (u <= prob[newTopic])
                break;
        }

        td.zk[newTopic] ++;
        td.nkt[newTopic][t1] ++;
        td.nkt[newTopic][t2] ++;
        td.nk[newTopic] += 2 ;
        z[d][n] = newTopic;
    }

    delete []prob;
}

void estimate(corpus &co, const char *fPhi, const char *fTheta) {
    int V = co.num_terms;
    int B = co.biterms.size();
    processor td;
    collect_nkt (co, td);

    ofstream op(fPhi);
    for (int k = 0; k < topic; k ++){
        for (int t = 0; t < V; t ++){
            double phi = ((td.nkt[k][t] + beta) / (td.nk[k] + V * beta));
            op << phi << " ";
        }
        op << "\n";
    }
    op.close();
                                                                    
    ofstream ot(fTheta);
    for (int k = 0; k < topic; k ++){
        double theta = ((td.zk[k] + alpha) / (B + topic * alpha));
        ot << theta << " ";
    }
    ot.close();
}

void runBTM(corpus &co,int iter) {
    int cpu_num = omp_get_num_procs();
    int doc_num = co.docs.size();
    vector<pair<int,int> > blocks;
    int step = doc_num / cpu_num;
    for (int i = 0; i < cpu_num; ++i) {
        int begin = i * step;
        int end = (i + 1) * step;
        blocks.push_back(make_pair(begin, end));
    }
    blocks[cpu_num - 1].second += doc_num % cpu_num;

    initializeModel(co);

    for (int i = 0; i < iter; ++i) {
        processor *td = new processor[cpu_num];

        #pragma omp parallel for
        for (int p = 0; p < cpu_num; ++p) {
            collect_nkt (co, td[p]);
        }

        #pragma omp parallel for
        for (int j = 0; j < cpu_num; ++j) {
            for (int d = blocks[j].first; d < blocks[j].second; ++d) {
                inferenceModel(co, d, td[j]);
            }
        }

        delete []td;

        if ((i + 1) % 10 == 0) {
            char phi[100] = {0};
            char theta[100] = {0};
            sprintf(phi,"/data0/data/btm/phi%d",i);
            sprintf(theta,"/data0/data/btm/theta%d",i);
            estimate(co, phi, theta);
        }
    }
}

int main(int argc, char* argv[]) {
    const char *biterm = "/data0/data/btm/bitermIndex";
    const char *doc = "/data0/data/btm/corpus";
    const char *phi = "./phi";
    const char *theta = "./theta";
    int iter = 500;

    printf("begin start\n");

    struct corpus co;
    load_corpus(biterm,doc,co);

    runBTM(co,iter);

    estimate(co, phi, theta);

    return 1; 
}
