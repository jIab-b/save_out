#include "ggml.h"
#include "gguf.h"

#include <stdexcept>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <regex>
#include <cstdint>
#include <climits>

namespace sd {

struct Model {
    ggml_context * ctx = nullptr;
    gguf_context * uf = nullptr;
    std::unordered_map<std::string, ggml_tensor *> tensors;
    std::unordered_map<std::string, std::string> kv;
};

class Tokenizer {
public:
    static Tokenizer from_text_model(const Model & m);
    std::vector<int32_t> encode(const std::string & text, bool add_special=true) const;
    int32_t bos_id() const { return id_bos; }
    int32_t eos_id() const { return id_eos; }
    int32_t unk_id() const { return id_unk; }
private:
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> merge_ranks;
    int32_t id_bos = -1, id_eos = -1, id_unk = -1, id_pad = -1;
    static int32_t get_int_kv(const Model & m, const std::string & key, int32_t def);
    // helpers
    static std::string bytes_to_unicode(const std::string & s);
    static std::vector<std::string> clip_pretok(const std::string & text);
};

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
        std::set<std::pair<std::string, std::string>> pairs;
        for (size_t i = 0; i + 1 < tokens.size(); ++i) pairs.insert({tokens[i], tokens[i+1]});
        return pairs;
    };
    auto bpe = [&](const std::string & token) {
        // token is already byte-encoded unicode string
        if (merge_ranks.empty()) return std::vector<std::string>{token};
        std::vector<std::string> word;
        word.reserve(token.size());
        // split into unicode characters
        for (size_t i = 0; i < token.size(); ) {
            unsigned char c = (unsigned char)token[i];
            // treat as single byte char
            word.emplace_back(token.substr(i, 1));
            i += 1;
        }
        if (word.empty()) return word;
        while (true) {
            auto pairs = get_pairs(word);
            int best_rank = INT_MAX;
            std::pair<std::string,std::string> best_pair;
            for (const auto & p : pairs) {
                auto it = merge_ranks.find(p.first + " " + p.second);
                if (it != merge_ranks.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_pair = p;
                }
            }
            if (best_rank == INT_MAX) break;
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
    // CLIP regex-based pre-tokenization
    static const std::regex pat(
        "'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?\\d+| ?[^A-Za-z0-9\\s]+|\\s+(?!\\S)|\\s+",
        std::regex::optimize);
    std::string bt = bytes_to_unicode(text);
    auto words_begin = std::sregex_iterator(bt.begin(), bt.end(), pat);
    auto words_end = std::sregex_iterator();
    for (auto it = words_begin; it != words_end; ++it) {
        std::string w = it->str();
        // BPE over the unicode-escaped bytes
        auto chunks = bpe(w);
        for (const auto & ch : chunks) {
            auto jt = token_to_id.find(ch);
            if (jt != token_to_id.end()) {
                out.push_back(jt->second);
            } else if (id_unk >= 0) {
                out.push_back(id_unk);
            }
        }
    }
    if (add_special && id_eos >= 0) out.push_back(id_eos);
    return out;
}

// Map bytes 0..255 to a set of printable unicode chars (GPT-2/CLIP scheme)
std::string Tokenizer::bytes_to_unicode(const std::string & s) {
    static std::vector<unsigned> bs;
    static std::vector<unsigned> cs;
    static bool init = false;
    if (!init) {
        // Following OpenAI's byte encoder approach
        for (unsigned i = 33; i <= 126; ++i) bs.push_back(i);
        for (unsigned i = 161; i <= 172; ++i) bs.push_back(i);
        for (unsigned i = 174; i <= 255; ++i) bs.push_back(i);
        cs = bs;
        unsigned n = 0;
        for (unsigned b = 0; b < 256; ++b) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + n);
                ++n;
            }
        }
        init = true;
    }
    std::unordered_map<unsigned,unsigned> map;
    map.reserve(bs.size());
    for (size_t i = 0; i < bs.size(); ++i) map[bs[i]] = cs[i];
    std::string out;
    out.reserve(s.size()*2);
    for (unsigned char c : s) {
        unsigned m = map[(unsigned)c];
        // Encode as UTF-8 for codepoint m
        if (m <= 0x7F) out.push_back((char)m);
        else if (m <= 0x7FF) {
            out.push_back((char)(0xC0 | ((m >> 6) & 0x1F)));
            out.push_back((char)(0x80 | (m & 0x3F)));
        } else if (m <= 0xFFFF) {
            out.push_back((char)(0xE0 | ((m >> 12) & 0x0F)));
            out.push_back((char)(0x80 | ((m >> 6) & 0x3F)));
            out.push_back((char)(0x80 | (m & 0x3F)));
        } else {
            out.push_back((char)(0xF0 | ((m >> 18) & 0x07)));
            out.push_back((char)(0x80 | ((m >> 12) & 0x3F)));
            out.push_back((char)(0x80 | ((m >> 6) & 0x3F)));
            out.push_back((char)(0x80 | (m & 0x3F)));
        }
    }
    return out;
}

}
