// Copyright 2013 lijiankou. All Rights Reserved.
// Author: lijk_start@163.com (Jiankou Li)
#ifndef ML_DOCUMENT_H_
#define ML_DOCUMENT_H_
#include "base_head.h"
namespace ml {
struct Document {
  VInt words;
  VInt counts;
  size_t total;
  Document() : total(0) {}
  inline size_t ULen() const { return words.size();}
  inline size_t TLen() const { return total;}
};

typedef std::vector<Document> VDocument;
typedef const Document DocumentC;

struct Corpus {
  VDocument docs;
  int num_terms;  // max index of words
  size_t t_words_num; // num of all terms in corpus;
  Corpus() : num_terms(0), t_words_num(0) {}
  size_t Len() const { return docs.size();}
  int TermNum() const { return num_terms;}
  size_t TWordsNum() const { return t_words_num;}
  size_t TLen(int d) const { return docs[d].TLen();}
  size_t ULen(int d) const { return docs[d].ULen();}
  void ULen(VInt* v) const;

  int Word(int d, int n) const { return docs[d].words[n];}
  int Count(int d, int n) const { return docs[d].counts[n];}

  void LoadData(const Str &filename);
  size_t MaxCorpusLen() const;
  void RandomOrder();

  void NewLatent(VVInt* z) const;
  void NewLatent(VVReal* z) const;
  void NewLatent(VVVReal* z, int k) const;
  void UpdateTWordsNum();
};

typedef const Corpus CorpusC;

void SplitData(const Corpus &c, double value, Corpus* train, Corpus* test);
}  // namespace ml 
#endif// ML_DOCUMENT_H_
