#include "../include/encoding.hpp"

namespace encodings {
    std::uint8_t encodeByte5b6b(std::uint8_t input, int RD) {
        if (RD == -1)
            return CODE_5b6b.at(input).first;
        else if (RD == 1)
            return CODE_5b6b.at(input).second;
        else
            throw std::invalid_argument("RD must be -1 or 1");
    }

    std::uint8_t encodeByte3b4b(std::uint8_t input, int RD) {
        if (RD == -1)
            return CODE_3b4b.at(input).first;
        else if (RD == 1)
            return CODE_3b4b.at(input).second;
        else
            throw std::invalid_argument("RD must be -1 or 1");
    }

    int count_ones(symbol10 input) {
        int count = 0;
        for (int i = 0; i < 8; i++) {
            count += (input >> i) & 1;
        }
        return count;
    }

    symbol10 encodeByte8b10b(std::uint8_t input, int RD) {
        std::uint8_t x = input & EDCBA;
        std::uint8_t y = input >> 5;
        std::uint8_t abcdei = encodeByte5b6b(x, RD);
        if (y == 7) {                                               // check for special case of D.x.P7/D.x.A7
            if ((RD == -1 && (x == 17 || x == 18 || x == 20)) ||    // see https://en.wikipedia.org/wiki/8b/10b_encoding for explanation
                (RD ==  1 && (x == 11 || x == 13 || x == 24))) {
                    y = 8;
                }
        }
        std::uint8_t fghj = encodeByte3b4b(y, RD);
        symbol10 output = 0;
        output = (abcdei << 4) | fghj;
        return output;
    }

    bytesVec encodeBytesVec8b10b(const bytesVec& data) {
        if (data.size() % 4 != 0) {
            throw std::invalid_argument("Data size must be divisible by 4 " + std::to_string(data.size()));
        }

        int RD = -1;
        bytesVec encoded;               // encoded data, in a vector of bytes (bytesVec)
        std::uint64_t buffer = 0;       // buffer used to hold 4 10bit symbols before they are saved to encoded bytesVec (as 5 bytes)
        int bitsInBuffer = 0;           // state of the buffer
        symbol10 symbol;                // 10bit symbol to encode
        std::uint8_t invertBuffer[5] ;  // buffer used to invert the order of bytes in the output vector       

        for (int i = 0; i < data.size(); i++) {
            // encode new symbol and update buffer
            try {
                symbol = encodeByte8b10b(data[i], RD);
            } catch (std::out_of_range& e) {
                //std::cout << "Invalid input byte: " << std::to_string(data[i]) << std::endl;
                return {};
            } catch (std::invalid_argument& e) {
                //std::cout << "Invalid RD value: " << std::to_string(RD) << std::endl;
                return {};
            }
            buffer = (buffer << 10) | symbol;
            bitsInBuffer += 10;

            // check for running disparity and update RD if needed
            // '1's in symbol10 (uint16_t really) is counted by count_ones()
            // '0's in uint16_t would be 16 - count_ones(),
            // but we are looking for the difference in a 10bit symbol, so 
            // '0's in symbol10 is 16 - count_ones() - 6 (6 '0's of front-padding)
            // so the difference is '1's - '0's = count_ones() - (16 - count_ones() - 6) = 2*count_ones() - 10
            int diff = 2 * count_ones(symbol) - 10;
            if (diff < 0)
                RD = -1;
            else if (diff > 0)
                RD = 1;

            // if there are 40 bits (4 symbols 10bit each or 5 bytes) in the buffer, save them to encoded bytesVec
            // the invertBuffer array is used to invert the order of bytes coming into the final buffer
            if (bitsInBuffer == 40) {
                for (int j = 0; j < 5; j++) {
                    invertBuffer[j] = buffer & 0xFF;
                    buffer = buffer >> 8;
                    bitsInBuffer -= 8;
                }
                for (int j = 4; j >= 0; j--) {
                    encoded.push_back(invertBuffer[j]);
                }
                bitsInBuffer = 0;
                buffer = 0;
            }
        }

        return encoded;
    }

    std::uint8_t decodeSymbol10b8b(symbol10 symbol) {
        std::uint8_t abcdei = symbol >> 4;
        std::uint8_t fghj = symbol & 0b1111;
        std::uint8_t x=0, y=0;
        try {
            x = DECODE_5b6b.at(abcdei);
        } catch (std::out_of_range& e) {
        }
        try {
            y = DECODE_3b4b.at(fghj);
        } catch (std::out_of_range& e) {
        }

        return (y << 5) | x;
    }

    bytesVec decodeBytesVec10b8b(const bytesVec& data) {
        if (data.size() % 5 != 0) {
            throw std::invalid_argument("Data size must be divisible by 4");
        }

        int RD = -1;
        bytesVec decoded;               // decoded data, in a vector of bytes (bytesVec)
        std::uint64_t buffer = 0;       // buffer used to hold 5 bytes (4 10 bit symbols) before they are decoded and saved to decoded bytesVec
        int bitsInBuffer = 0;           // state of the buffer
        symbol10 symbol;                // 10bit symbol to decode
        std::uint8_t invertBuffer[4];   // buffer used to invert the order of bytes in the output vector

        for (int i = 0; i < data.size(); i++) {
            // push bytes into buffer
            buffer = (buffer << 8) | data[i];
            bitsInBuffer += 8;
            
            // after there are 40 bits (5 bytes, 4 symbols 10bit each) in the buffer
            // decode each symbol and save decoded bytes to decoded bytesVec
            if (bitsInBuffer == 40) {
                for (int j = 0; j < 4; j++) {
                    symbol = 0;
                    symbol = buffer & 0b0000001111111111;
                    buffer = buffer >> 10;
                    bitsInBuffer -= 10;
                    invertBuffer[j] = decodeSymbol10b8b(symbol);
                }
                for (int j = 3; j >= 0; j--) {
                    decoded.push_back(invertBuffer[j]);
                }
            }
        }

        return decoded;
    }
};