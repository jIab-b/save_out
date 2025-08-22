#include <string>
#include <cstdio>
#include <vector>

namespace sd {

bool write_ppm(const std::string & path, int w, int h, const std::vector<unsigned char> & rgb) {
    if ((int)rgb.size() < w * h * 3) return false;
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    std::fprintf(f, "P6\n%d %d\n255\n", w, h);
    const size_t n = (size_t)w * h * 3;
    const bool ok = std::fwrite(rgb.data(), 1, n, f) == n;
    std::fclose(f);
    return ok;
}

static void write_be32(std::vector<unsigned char> & out, unsigned v) {
    out.push_back((v >> 24) & 0xFF);
    out.push_back((v >> 16) & 0xFF);
    out.push_back((v >> 8) & 0xFF);
    out.push_back(v & 0xFF);
}

static unsigned crc32(const unsigned char * data, size_t len) {
    static unsigned table[256];
    static bool init = false;
    if (!init) {
        for (unsigned i = 0; i < 256; ++i) {
            unsigned c = i;
            for (int k = 0; k < 8; ++k) c = c & 1 ? 0xEDB88320u ^ (c >> 1) : (c >> 1);
            table[i] = c;
        }
        init = true;
    }
    unsigned c = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i) c = table[(c ^ data[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

static unsigned adler32(const unsigned char * data, size_t len) {
    unsigned a = 1, b = 0;
    for (size_t i = 0; i < len; ++i) { a = (a + data[i]) % 65521; b = (b + a) % 65521; }
    return (b << 16) | a;
}

bool write_png(const std::string & path, int w, int h, const std::vector<unsigned char> & rgb) {
    if (w <= 0 || h <= 0) return false;
    if ((int)rgb.size() < w * h * 3) return false;

    std::vector<unsigned char> png;
    const unsigned char sig[8] = {137,80,78,71,13,10,26,10};
    png.insert(png.end(), sig, sig+8);

    // IHDR
    std::vector<unsigned char> ihdr;
    write_be32(ihdr, (unsigned)w);
    write_be32(ihdr, (unsigned)h);
    ihdr.push_back(8);
    ihdr.push_back(2);
    ihdr.push_back(0);
    ihdr.push_back(0);
    ihdr.push_back(0);
    write_be32(png, ihdr.size());
    png.push_back('I'); png.push_back('H'); png.push_back('D'); png.push_back('R');
    png.insert(png.end(), ihdr.begin(), ihdr.end());
    unsigned cihdr = crc32(&png[png.size()-ihdr.size()-4], ihdr.size()+4);
    write_be32(png, cihdr);

    // IDAT (zlib uncompressed blocks)
    std::vector<unsigned char> raw;
    raw.reserve((size_t)(w*3+1)*h);
    for (int y = 0; y < h; ++y) {
        raw.push_back(0);
        const unsigned char * row = &rgb[(size_t)y*w*3];
        raw.insert(raw.end(), row, row + (size_t)w*3);
    }
    std::vector<unsigned char> z;
    z.push_back(0x78); z.push_back(0x01);
    size_t pos = 0;
    while (pos < raw.size()) {
        size_t chunk = std::min((size_t)65535, raw.size() - pos);
        bool last = (pos + chunk) == raw.size();
        z.push_back(last ? 1 : 0);
        z.push_back((unsigned char)(chunk & 0xFF));
        z.push_back((unsigned char)((chunk >> 8) & 0xFF));
        unsigned nlen = 0xFFFF - (unsigned)chunk;
        z.push_back((unsigned char)(nlen & 0xFF));
        z.push_back((unsigned char)((nlen >> 8) & 0xFF));
        z.insert(z.end(), raw.begin() + pos, raw.begin() + pos + chunk);
        pos += chunk;
    }
    unsigned ad = adler32(raw.data(), raw.size());
    write_be32(z, ad);

    write_be32(png, z.size());
    png.push_back('I'); png.push_back('D'); png.push_back('A'); png.push_back('T');
    png.insert(png.end(), z.begin(), z.end());
    unsigned cidat = crc32(&png[png.size()-z.size()-4], z.size()+4);
    write_be32(png, cidat);

    // IEND
    write_be32(png, 0);
    png.push_back('I'); png.push_back('E'); png.push_back('N'); png.push_back('D');
    unsigned ciend = crc32((const unsigned char *)"IEND", 4);
    write_be32(png, ciend);

    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const bool ok = std::fwrite(png.data(), 1, png.size(), f) == png.size();
    std::fclose(f);
    return ok;
}

}


