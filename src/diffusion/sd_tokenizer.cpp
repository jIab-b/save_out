#include "sd_tokenizer.h"

#include "gguf.h"

#include <stdexcept>
#include <unordered_set>
#include <sstream>

namespace sd {

int32_t Tokenizer::get_int_kv(const Model & m, const std::string & key, int32_t def) {
    auto it = m.kv.find(key);
    if (it == m.kv.end()) return def;
    try { return std::stoi(it->second); } catch (...) { return def; }
}

Tokenizer Tokenizer::from_text_model(const Model & m) {
    Tokenizer t;
    auto uf = m.uf;
    const int token_idx = gguf_find_key(uf, "tokenizer.ggml.tokens");
    if (token_idx == -1) throw std::runtime_error("tokenizer tokens not found");
    const int n_tokens = gguf_get_arr_n(uf, token_idx);
    t.id_to_token.reserve(n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        std::string tok = gguf_get_arr_str(uf, token_idx, i);
        t.token_to_id.emplace(tok, i);
        t.id_to_token.emplace_back(std::move(tok));
    }
    const int merges_idx = gguf_find_key(uf, "tokenizer.ggml.merges");
    if (merges_idx != -1) {
        const int n_merges = gguf_get_arr_n(uf, merges_idx);
        for (int i = 0; i < n_merges; ++i) {
            std::string mrg = gguf_get_arr_str(uf, merges_idx, i);
            t.merge_ranks.emplace(mrg, i);
        }
    }
    t.id_bos = get_int_kv(m, "tokenizer.ggml.bos_token_id", -1);
    t.id_eos = get_int_kv(m, "tokenizer.ggml.eos_token_id", -1);
    t.id_unk = get_int_kv(m, "tokenizer.ggml.unknown_token_id", -1);
    t.id_pad = get_int_kv(m, "tokenizer.ggml.padding_token_id", -1);
    return t;
}

std::vector<int32_t> Tokenizer::encode(const std::string & text, bool add_special) const {
    auto get_pairs = [](const std::vector<std::string> & tokens) {
        std::unordered_set<std::pair<std::string,std::string>,
            std::hash<std::string>,
            bool(*)(const std::pair<std::string,std::string>&, const std::pair<std::string,std::string>&)> pairs(0, std::hash<std::string>(),
            [](const auto & a, const auto & b){ return a.first==b.first && a.second==b.second; });
        for (size_t i = 0; i + 1 < tokens.size(); ++i) {
            pairs.insert({tokens[i], tokens[i+1]});
        }
        return pairs;
    };

    auto bpe = [&](const std::string & token) {
        std::vector<std::string> word;
        word.reserve(token.size());
        for (unsigned char c : token) word.emplace_back(std::string(1, (char)c));
        if (word.empty()) return word;
        while (true) {
            auto pairs = get_pairs(word);
            int best_rank = INT32_MAX;
            std::pair<std::string,std::string> best_pair;
            for (const auto & p : pairs) {
                auto it = merge_ranks.find(p.first + " " + p.second);
                if (it != merge_ranks.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_pair = p;
                }
            }
            if (best_rank == INT32_MAX) break;
            std::vector<std::string> new_word;
            new_word.reserve(word.size());
            for (size_t i = 0; i < word.size(); ) {
                if (i + 1 < word.size() && word[i] == best_pair.first && word[i+1] == best_pair.second) {
                    new_word.emplace_back(word[i] + word[i+1]);
                    i += 2;
                } else {
                    new_word.emplace_back(word[i]);
                    i += 1;
                }
            }
            word.swap(new_word);
        }
        return word;
    };

    std::vector<int32_t> out;
    if (add_special && id_bos >= 0) out.push_back(id_bos);
    std::istringstream iss(text);
    std::string tok;
    while (iss >> tok) {
        auto chunks = bpe(tok);
        for (const auto & ch : chunks) {
            auto it = token_to_id.find(ch);
            if (it != token_to_id.end()) {
                out.push_back(it->second);
            } else if (id_unk >= 0) {
                out.push_back(id_unk);
            }
        }
    }
    if (add_special && id_eos >= 0) out.push_back(id_eos);
    return out;
}

}


