// Author: Marek Szymański
// Description: 

#pragma once

#include <vector>
#include <cstdint>
#include <stdexcept>

#include "encodingtables.h"

using byte = std::uint8_t;
using bytesVec = std::vector<byte>;
using symbol10 = std::uint16_t;

namespace encodings {
    /**
     * @brief Bit mask to extract the younger 5 bits from byte
     * 
     */
    const std::uint8_t EDCBA = 0b00011111;
    /**
     * @brief Bit mask to extract the older 3 bits from byte
     * 
     */
    const std::uint8_t HGF   = 0b11100000;

    /**
     * @brief 5b/6b encoding of a single symbol
     * @details Encodes 5 bits to 6 bits, using the 5b/6b encoding table, with th respect of running disparity (used in 8b/10b encoding)
     *          The table, defined in encodingtables.h, is a map of pairs of 6bit symbols, former for RD=-1 and latter for RD=1
     *          As it's not really feasible to pass around packets of 5 or 6 bits, the function takes 8bit input and returns 8bit output
     *          the "additional" bits are filled with '0's 
     *          Throws std::invalid_argument if RD is neither -1 nor 1
     * 
     * @param input         - symbol to encode - uint8_t used to transport 5 bit symbol of 5b/6b encoding, with the "additional" bits are set to 0
     * @param RD            - running disparity, -1 or 1
     * @return std::uint8_t - encoded symbol - uint8_t used to transport 6 bit symbol of 5b/6b encoding, with the "additional" bits are set to 0
     */
    std::uint8_t encodeByte5b6b(std::uint8_t input, int RD);

    /**
     * @brief 3b/4b encoding of a single symbol
     * @details Encodes 3 bits to 4 bits, using the 3b/4b encoding table, with th respect of running disparity (used in 8b/10b encoding)
     *          The table, defined in encodingtables.h, is a map of pairs of 4bit symbols, former for RD=-1 and latter for RD=1
     *          As it's not really feasible to pass around packets of 3 or 4 bits, the function takes 8bit input and returns 8bit output
     *          the "additional" bits are filled with '0's 
     *          Throws std::invalid_argument if RD is neither -1 nor 1
     * 
     * @param input         - symbol to encode - uint8_t used to transport 3 bit symbol of 3b/4b encoding, with the "additional" bits are set to 0
     * @param RD            - running disparity, -1 or 1
     * @return std::uint8_t - encoded symbol - uint8_t used to transport 4 bit symbol of 3b/4b encoding, with the "additional" bits are set to 0
     */
    std::uint8_t encodeByte3b4b(std::uint8_t input, int RD);

    /**
     * @brief Counts high bits ('1') in the given 10bit symbol
     * @details While such implementation is obviously inefficient, it is used to avoid relying on compiler's built-in functions
     * 
     * @param input - 10bit symbol to count high bits in
     * @return int  - number of high bits in the symbol
     */
    int count_ones(symbol10 input);

    /**
     * @brief 8b/10b encoding of a single symbol
     * @details Encodes 8 bits to 10 bits, using 5b/6b and 3b/4b encodings
     *          See https://en.wikipedia.org/wiki/8b/10b_encoding for more detailed explanation of the algorithm
     *          Uses encodeByte5b6b() to encode the younger 5 bits of the input symbol into the older 6 bits of the output symbol
     *          and encodeByte3b4b() to encode the older 3 bits of the input symbol into the younger 4 bits of the output symbol
     *          In order to avoid long series of 0 or 1 the 8b/10b encoding tracks the "running disparity" (RD) 
     *          RD is the difference between the number of '1' and '0' bits in the previous encoded symbol
     *          and is used to choose the correct 5b/6b and 3b/4b encoding of the next symbol
     * 
     * @param input     - symbol to encode - uint8_t used to transport 8 bit symbol of 8b/10b encoding, with the "additional" bits are set to 0
     * @param RD        - running disparity, -1 or 1
     * @return symbol10 - uint16_t used to transport 10 bit symbol of 8b/10b encoding, with the "additional" bits are set to 0
     */
    symbol10 encodeByte8b10b(std::uint8_t input, int RD);

    /**
     * @brief Encodes vector of bytes to 8b/10b encoded vector of bytes, the given vector MUST have length divisible by 4
     * @details Encodes gicen vector of bytes (bytesVec) with the 8b/10b encoding
     *          see https://en.wikipedia.org/wiki/8b/10b_encoding for more detailed explanation
     *          see encodeByte8b10b() for how single symbol is encoded
     *          The 10 bit encoded symbols are buffered until 40 bits (4 symbols) are stored
     *          so that 5 bytes can be written to output vector at once
     *          (the bytes have to be written in reverse order, as the 10bit symbols are stored in reverse order in the buffer)
     *          due to this the vector being encoded must have length divisible by 4 
     *          throws std::invalid_argument otherwise.
     *          The function also tracks the "running disparity" (RD) of the encoded symbols
     *          and updates it accordingly
     * 
     * @param data      - vector of bytes to encode, must have length divisible by 4, not modified
     * @return bytesVec - 8b/10b encoded vector of bytes
     */
    bytesVec encodeBytesVec8b10b(const bytesVec& data);

    /**
     * @brief Decodes 10bit symbol to 8bit symbol
     * @details Decodes 10bit symbol to 8bit symbol, using 5b/6b and 3b/4b decodings
     *          See https://en.wikipedia.org/wiki/8b/10b_encoding for more detailed explanation of the algorithm
     *          Uses DECODE_5b6b and DECODE_3b4b tables to decode the 10bit symbol
     *          The function first extracts the younger 6 bits of the 10bit symbol and decodes them using DECODE_5b6b table
     *          Then it extracts the older 4 bits of the 10bit symbol and decodes them using DECODE_3b4b table
     *          The decoded 6 and 4 bits are then combined into 8bit symbol
     * 
     * @param symbol        - 10bit symbol to decode
     * @return std::uint8_t - 8bit decoded symbol
     */
    std::uint8_t decodeSymbol10b8b(symbol10 symbol);

    /**
     * @brief Decode vector of bytes from 8b/10b encoded vector of bytes, the given vector MUST have length divisible by 5
     * @details Decodes given vector of bytes (bytesVec) with the 8b/10b encoding
     *          see decodeSymbol10b8b() for how single symbol is decoded
     *          The 10 bit encoded symbols are buffered until 40 bits (4 symbols) are stored
     *          so that 5 bytes can be written to output vector at once
     *          (the bytes have to be written in reverse order, as the 10bit symbols are stored in reverse order in the buffer)
     *          due to this the vector being decoded must have length divisible by 5
     *          throws std::invalid_argument otherwise.
     *          During decoding 8b/10b encoding there is no need to track the "running disparity" (RD) of the encoded symbols
     * 
     * @param data      - vector of bytes encided in 8b/10b to decode, must have length divisible by 4, not modified
     * @return bytesVec - decoded vector of bytes
     */
    bytesVec decodeBytesVec10b8b(const bytesVec& data);
};