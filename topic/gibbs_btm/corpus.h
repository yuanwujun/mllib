// Copyright 2014 yuanwujun. All Rights Reserved.
// Author: real.yuanwj@gmail.com

#ifndef _CORPUS_H_
#define _CORPUS_H_

#include <vector>

using namespace std;

struct corpus {
    vector<pair<int,int> > biterms;
    vector<vector<int> > docs;

    int num_terms;
};

int load_corpus(const char *file_biterm, const char *file_doc, corpus &cor);

#endif // _CORPUS_H_
