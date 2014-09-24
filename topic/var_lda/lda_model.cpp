#include "lda_model.h"

#include "base_head.h"

namespace ml {
const int NUM_INIT = 1;
void LdaMLE(int estimate_alpha, LdaSuffStatsC &ss, LdaModel* m) {
  for (int k = 0; k < m->num_topics; k++) {
    for (int w = 0; w < m->num_terms; w++) {
      if (ss.class_word[k][w] > 0) {
        m->log_prob_w[k][w] = log(ss.class_word[k][w]) - log(ss.class_total[k]);
      } else {
        m->log_prob_w[k][w] = -100;
      }
    }
  }
  if (estimate_alpha == 1) {
    m->alpha = OptAlpha(ss.alpha_suffstats, ss.num_docs, m->num_topics);
  }
}

void NewLdaModel(int num_topics, int num_terms, LdaModel* model) {
  model->num_topics = num_topics;
  model->num_terms = num_terms;
  model->alpha = 1.0;
  model->log_prob_w = NewArray(num_topics, num_terms);
  Init(num_topics, num_terms, 0.0, model->log_prob_w);
}

void NewLdaSuffStats(LdaModelC &m, LdaSuffStats* ss) {
  ss->class_total = new double[m.num_topics];
  double init = 0.0;
  Init(m.num_topics, init, ss->class_total);
  ss->class_word = NewArray(m.num_topics, m.num_terms);
  Init(m.num_topics, m.num_terms, init, ss->class_word);
}

void InitSS(LdaModelC &model, double value, LdaSuffStats* ss) {
  for (int k = 0; k < model.num_topics; k++) {
    ss->class_total[k] = value;
    for (int w = 0; w < model.num_terms; w++) {
      ss->class_word[k][w] = value;
    }
  }
  ss->num_docs = value;
  ss->alpha_suffstats = value;
}

void RandomInitSS(const LdaModel &m, LdaSuffStats* ss) {
  for (int k = 0; k < m.num_topics; k++) {
    for (int n = 0; n < m.num_terms; n++) {
      ss->class_word[k][n] += 1.0 / m.num_terms + myrand();
      ss->class_total[k] += ss->class_word[k][n];
    }
  }
}

void CorpusInitSS(const Corpus &c, const LdaModel &m, LdaSuffStats* ss) {
  for (int k = 0; k < m.num_topics; k++) {
    for (int i = 0; i < NUM_INIT; i++) {
      const Document &doc =
       c.docs[static_cast<int>(floor(myrand() * c.docs.size()))];
      for (size_t n = 0; n < doc.words.size(); n++) {
        ss->class_word[k][doc.words[n]] += doc.counts[n];
      }
    }
    for (int n = 0; n < m.num_terms; n++) {
      ss->class_word[k][n] += 1.0;
      ss->class_total[k] = ss->class_total[k] + ss->class_word[k][n];
    }
  }
}

double OptAlpha(double ss, int d, int k) {
  const double  NEWTON_THRESH = 1e-5;
  const int MAX_ALPHA_ITER = 1000;
  double init_a = 100;
  double log_a = log(init_a);
  int iter = 0;
  double df = 0;
  do {
    iter++;
    double a = exp(log_a);
    if (isnan(a)) {
      init_a = init_a * 10;
      a = init_a;
      log_a = log(a);
    }
    df = DAlhood(a, ss, d, k);
    double d2f = D2Alhood(a, d, k);
    log_a = log_a - df / (d2f * a + df);
  } while ((fabs(df) > NEWTON_THRESH) && (iter < MAX_ALPHA_ITER));
  return exp(log_a);
}
}  // namespace ml
