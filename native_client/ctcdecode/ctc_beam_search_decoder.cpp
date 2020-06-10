#include "ctc_beam_search_decoder.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <utility>

#include "decoder_utils.h"
#include "ThreadPool.h"
#include "fst/fstlib.h"
#include "path_trie.h"

// helper function for kws_init
void setup_labels(const std::vector<unsigned int>& labels,
                 const int blank,
                 std::vector<int>& labels_w_blanks) {

  for (int i = 0; i < labels.size(); ++i) {
    labels_w_blanks.push_back(blank);
    labels_w_blanks.push_back(labels[i]);
  }
  labels_w_blanks.push_back(blank);
}

static const float neginf = -std::numeric_limits<float>::infinity();

// helper function for kws_next
inline float log_add(float a, float b) {
  if (a == neginf) return b;
  if (b == neginf) return a;
  if (a > b)
    return log1p(exp(b-a)) + a;
  else
    return log1p(exp(a-b)) + b;
}

int
DecoderState::init(const Alphabet& alphabet,
                   size_t beam_size,
                   double cutoff_prob,
                   size_t cutoff_top_n,
                   std::shared_ptr<Scorer> ext_scorer)
{
  // assign special ids
  abs_time_step_ = 0;
  space_id_ = alphabet.GetSpaceLabel();
  blank_id_ = alphabet.GetSize();

  beam_size_ = beam_size;
  cutoff_prob_ = cutoff_prob;
  cutoff_top_n_ = cutoff_top_n;
  ext_scorer_ = ext_scorer;
  start_expanding_ = false;

  // init prefixes' root
  PathTrie *root = new PathTrie;
  root->score = root->log_prob_b_prev = 0.0;
  prefix_root_.reset(root);
  prefixes_.push_back(root);

  if (ext_scorer && (bool)(ext_scorer_->dictionary)) {
    // no need for std::make_shared<>() since Copy() does 'new' behind the doors
    auto dict_ptr = std::shared_ptr<PathTrie::FstType>(ext_scorer->dictionary->Copy(true));
    root->set_dictionary(dict_ptr);
    auto matcher = std::make_shared<fst::SortedMatcher<PathTrie::FstType>>(*dict_ptr, fst::MATCH_INPUT);
    root->set_matcher(matcher);
  }

  return 0;
}

int DecoderState::kws_init(const Alphabet& alphabet,
                           const char* labels)
{
  assert(strlen(labels) > 0); // empty keyword is not supported
  abs_time_step_ = 0;
  blank_id_ = alphabet.GetSize();
  setup_labels(alphabet.StringToLabels(labels), blank_id_, labels_w_blanks);
  S = labels_w_blanks.size();
  alphas.resize(S, neginf);

  return 0;
}

void
DecoderState::next(const double *probs,
                   int time_dim,
                   int class_dim)
{
  // prefix search over time
  for (size_t rel_time_step = 0; rel_time_step < time_dim; ++rel_time_step, ++abs_time_step_) {
    auto *prob = &probs[rel_time_step*class_dim];

    // At the start of the decoding process, we delay beam expansion so that
    // timings on the first letters is not incorrect. As soon as we see a
    // timestep with blank probability lower than 0.999, we start expanding
    // beams.
    if (prob[blank_id_] < 0.999) {
      start_expanding_ = true;
    }

    // If not expanding yet, just continue to next timestep.
    if (!start_expanding_) {
      continue;
    }

    float min_cutoff = -NUM_FLT_INF;
    bool full_beam = false;
    if (ext_scorer_) {
      size_t num_prefixes = std::min(prefixes_.size(), beam_size_);
      std::partial_sort(prefixes_.begin(),
                        prefixes_.begin() + num_prefixes,
                        prefixes_.end(),
                        prefix_compare);

      min_cutoff = prefixes_[num_prefixes - 1]->score +
                   std::log(prob[blank_id_]) - std::max(0.0, ext_scorer_->beta);
      full_beam = (num_prefixes == beam_size_);
    }

    std::vector<std::pair<size_t, float>> log_prob_idx =
        get_pruned_log_probs(prob, class_dim, cutoff_prob_, cutoff_top_n_);
    // loop over class dim
    for (size_t index = 0; index < log_prob_idx.size(); index++) {
      auto c = log_prob_idx[index].first;
      auto log_prob_c = log_prob_idx[index].second;

      for (size_t i = 0; i < prefixes_.size() && i < beam_size_; ++i) {
        auto prefix = prefixes_[i];
        if (full_beam && log_prob_c + prefix->score < min_cutoff) {
          break;
        }

        // blank
        if (c == blank_id_) {
          prefix->log_prob_b_cur =
              log_sum_exp(prefix->log_prob_b_cur, log_prob_c + prefix->score);
          continue;
        }

        // repeated character
        if (c == prefix->character) {
          prefix->log_prob_nb_cur = log_sum_exp(
              prefix->log_prob_nb_cur, log_prob_c + prefix->log_prob_nb_prev);
        }

        // get new prefix
        auto prefix_new = prefix->get_path_trie(c, abs_time_step_, log_prob_c);

        if (prefix_new != nullptr) {
          float log_p = -NUM_FLT_INF;

          if (c == prefix->character &&
              prefix->log_prob_b_prev > -NUM_FLT_INF) {
            log_p = log_prob_c + prefix->log_prob_b_prev;
          } else if (c != prefix->character) {
            log_p = log_prob_c + prefix->score;
          }

          if (ext_scorer_) {
            // skip scoring the space in word based LMs
            PathTrie* prefix_to_score;
            if (ext_scorer_->is_utf8_mode()) {
              prefix_to_score = prefix_new;
            } else {
              prefix_to_score = prefix;
            }

            // language model scoring
            if (ext_scorer_->is_scoring_boundary(prefix_to_score, c)) {
              float score = 0.0;
              std::vector<std::string> ngram;
              ngram = ext_scorer_->make_ngram(prefix_to_score);
              bool bos = ngram.size() < ext_scorer_->get_max_order();
              score = ext_scorer_->get_log_cond_prob(ngram, bos) * ext_scorer_->alpha;
              log_p += score;
              log_p += ext_scorer_->beta;
            }
          }

          prefix_new->log_prob_nb_cur =
              log_sum_exp(prefix_new->log_prob_nb_cur, log_p);
        }
      }  // end of loop over prefix
    }    // end of loop over alphabet

    // update log probs
    prefixes_.clear();
    prefix_root_->iterate_to_vec(prefixes_);

    // only preserve top beam_size prefixes
    if (prefixes_.size() > beam_size_) {
      std::nth_element(prefixes_.begin(),
                       prefixes_.begin() + beam_size_,
                       prefixes_.end(),
                       prefix_compare);
      for (size_t i = beam_size_; i < prefixes_.size(); ++i) {
        prefixes_[i]->remove();
      }

      // Remove the elements from std::vector
      prefixes_.resize(beam_size_);
    }
  }  // end of loop over time
}

void DecoderState::kws_next(const double* probs,
                            const int T,
                            const int class_dim)
{
  int time_start = 0;
  if (abs_time_step_ == 0) {
    alphas[0] = std::log(1 - probs[labels_w_blanks[1]]);
    alphas[1] = std::log(probs[labels_w_blanks[1]]);
    time_start = 1;
  }

  std::vector<float> next_alphas;
  next_alphas.resize(S, neginf);

  for(int t = time_start; t < T; ++t, ++abs_time_step_) {
    const double *prob = &probs[t * class_dim];
    std::fill(next_alphas.begin(), next_alphas.end(), neginf);

    for(int s = 0; s < S; ++s) {
      const int l = labels_w_blanks[s];

      float p;
      if (s == 0) {
        p = std::log(1 - prob[labels_w_blanks[1]]);
      } else if (s == (S - 1)) {
        p = std::log(1 - prob[labels_w_blanks[S-2]]);
      } else {
        p = std::log(prob[l]);
      }

      if (s > 1 && l != blank_id_ && l != labels_w_blanks[s-2]) {
        next_alphas[s] = log_add(alphas[s], alphas[s-1]);
        next_alphas[s] = log_add(next_alphas[s], alphas[s-2]);
        next_alphas[s] += p;
      } else if (s > 0) {
        next_alphas[s] = log_add(alphas[s], alphas[s-1]);
        next_alphas[s] += p;
      } else {
        next_alphas[s] = alphas[s];
        next_alphas[s] += p;
      }
    }

    std::swap(alphas, next_alphas);
  }
}

std::vector<Output>
DecoderState::decode(size_t num_results) const
{
  std::vector<PathTrie*> prefixes_copy = prefixes_;
  std::unordered_map<const PathTrie*, float> scores;
  for (PathTrie* prefix : prefixes_copy) {
    scores[prefix] = prefix->score;
  }

  // score the last word of each prefix that doesn't end with space
  if (ext_scorer_) {
    for (size_t i = 0; i < beam_size_ && i < prefixes_copy.size(); ++i) {
      auto prefix = prefixes_copy[i];
      if (!ext_scorer_->is_scoring_boundary(prefix->parent, prefix->character)) {
        float score = 0.0;
        std::vector<std::string> ngram = ext_scorer_->make_ngram(prefix);
        bool bos = ngram.size() < ext_scorer_->get_max_order();
        score = ext_scorer_->get_log_cond_prob(ngram, bos) * ext_scorer_->alpha;
        score += ext_scorer_->beta;
        scores[prefix] += score;
      }
    }
  }

  using namespace std::placeholders;
  size_t num_returned = std::min(prefixes_copy.size(), num_results);
  std::partial_sort(prefixes_copy.begin(),
                    prefixes_copy.begin() + num_returned,
                    prefixes_copy.end(),
                    std::bind(prefix_compare_external, _1, _2, scores));

  std::vector<Output> outputs;
  outputs.reserve(num_returned);

  // compute aproximate ctc score as the return score, without affecting the
  // return order of decoding result. To delete when decoder gets stable.
  for (size_t i = 0; i < num_returned; ++i) {
    Output output;
    prefixes_copy[i]->get_path_vec(output.tokens, output.timesteps);
    float approx_ctc = scores[prefixes_copy[i]];
    if (ext_scorer_) {
      auto words = ext_scorer_->split_labels_into_scored_units(output.tokens);
      // remove term insertion weight
      approx_ctc -= words.size() * ext_scorer_->beta;
      // remove language model weight
      approx_ctc -= (ext_scorer_->get_sent_log_prob(words)) * ext_scorer_->alpha;
    }
    output.confidence = -approx_ctc;
    outputs.push_back(output);
  }

  return outputs;
}


std::vector<Output> DecoderState::kws_decode(size_t num_results) const
{
  Output output;
  output.confidence = log_add(alphas[S-2], alphas[S-1]);

  std::vector<Output> outputs;
  outputs.push_back(output);
  return outputs;
}

std::vector<Output> ctc_beam_search_decoder(
    const double *probs,
    int time_dim,
    int class_dim,
    const Alphabet &alphabet,
    size_t beam_size,
    double cutoff_prob,
    size_t cutoff_top_n,
    std::shared_ptr<Scorer> ext_scorer)
{
  DecoderState state;
  state.init(alphabet, beam_size, cutoff_prob, cutoff_top_n, ext_scorer);
  state.next(probs, time_dim, class_dim);
  return state.decode();
}

std::vector<Output> kws_decoder(
    const double* probs,
    int time_dim,
    int class_dim,
    const Alphabet &alphabet,
    const char* labels)
{
  DecoderState state;
  state.kws_init(alphabet, labels);
  state.kws_next(probs, time_dim, class_dim);
  return state.kws_decode();
}

std::vector<std::vector<Output>>
ctc_beam_search_decoder_batch(
    const double *probs,
    int batch_size,
    int time_dim,
    int class_dim,
    const int* seq_lengths,
    int seq_lengths_size,
    const Alphabet &alphabet,
    size_t beam_size,
    size_t num_processes,
    double cutoff_prob,
    size_t cutoff_top_n,
    std::shared_ptr<Scorer> ext_scorer)
{
  VALID_CHECK_GT(num_processes, 0, "num_processes must be nonnegative!");
  VALID_CHECK_EQ(batch_size, seq_lengths_size, "must have one sequence length per batch element");
  // thread pool
  ThreadPool pool(num_processes);

  // enqueue the tasks of decoding
  std::vector<std::future<std::vector<Output>>> res;
  for (size_t i = 0; i < batch_size; ++i) {
    res.emplace_back(pool.enqueue(ctc_beam_search_decoder,
                                  &probs[i*time_dim*class_dim],
                                  seq_lengths[i],
                                  class_dim,
                                  alphabet,
                                  beam_size,
                                  cutoff_prob,
                                  cutoff_top_n,
                                  ext_scorer));
  }

  // get decoding results
  std::vector<std::vector<Output>> batch_results;
  for (size_t i = 0; i < batch_size; ++i) {
    batch_results.emplace_back(res[i].get());
  }
  return batch_results;
}

std::vector<std::vector<Output>>
kws_decoder_batch(
    const double* probs,
    int batch_size,
    int time_dim,
    int class_dim,
    const int* seq_lengths,
    int seq_lengths_size,
    const Alphabet &alphabet,
    size_t num_processes,
    const char* labels)
{
  VALID_CHECK_GT(num_processes, 0, "num_processes must be nonnegative!");
  VALID_CHECK_EQ(batch_size, seq_lengths_size, "must have one sequence length per batch element");
  // thread pool
  ThreadPool pool(num_processes);

  // enqueue the tasks of decoding
  std::vector<std::future<std::vector<Output>>> res;
  for (size_t i = 0; i < batch_size; ++i) {
    res.emplace_back(pool.enqueue(kws_decoder,
                                  &probs[i*time_dim*class_dim],
                                  seq_lengths[i],
                                  class_dim,
                                  alphabet,
                                  labels));
  }
  // get decoding results
  std::vector<std::vector<Output>> batch_results;
  for (size_t i = 0; i < batch_size; ++i) {
    batch_results.emplace_back(res[i].get());
  }
  return batch_results;
}
