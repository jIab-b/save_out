#pragma once

#include "sd_loader.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace sd {

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
};

}


