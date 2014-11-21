// Copyright 2014 yuanwujun. All Rights Reserved.
// Author: real.yuanwj@gmail.com

#include <stdio.h>

#include "corpus.h"

int load_corpus(const char *file_biterm,const char *file_doc, corpus &cor) {
    FILE *fin = fopen(file_biterm, "r");
    int t1 = 0, t2 = 0, index = 0;
    cor.num_terms = 0;
    while(fscanf(fin, "%d %d %d", &t1, &t2, &index) > 0) {
        cor.biterms.push_back(make_pair(t1,t2));
        int t = max(t1,t2);
        if ( t >= cor.num_terms) {
            cor.num_terms = t + 1;
        }
    }
    fclose(fin);

    fin = fopen(file_doc, "r");
    int length = 0;
    while ((fscanf(fin, "%10d", &length) != EOF)) {
        vector<int> doc(length);
        for (int n = 0; n < length; n++) {
            int biterm = 0;
            if (fscanf(fin, "%10d", &biterm) == 0) {
                continue;
            }
            doc.push_back(biterm);
        }
        cor.docs.push_back(doc);
    }
    fclose(fin);

    return 1;
}
