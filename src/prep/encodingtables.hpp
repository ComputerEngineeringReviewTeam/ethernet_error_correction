// Author: Marek Szymański
// Description: Collection of encoding and decoding tables for 5b/6b and 3b/4b encoding schemes.

#pragma once

#include <cstdint>
#include <unordered_map>
#include <utility>

namespace encodings
{
    /**
     * @brief 5b/6b encoding table
     * @details This table is taken from https://en.wikipedia.org/wiki/8b/10b_encoding.
     *          The storage format is: {EDCBA, {abcdei: if RD == -1, abcdei: if RD == 1}},
     *          where EDCBA is the 5b input and abcdei is the 6b output.
     *          RD is the running disparity, which is the difference between the number of 1s and 0s in the previous 5b block.
     *          Each 5 bit / 6 bit code is extended into std::uint8_t to make it easier to use in the code.
     */
    const std::unordered_map<std::uint8_t, std::pair<std::uint8_t, std::uint8_t>> CODE_5b6b = {
        // 0 - 15
        {0b0000000, {0b00100111, 0b00011000}},
        {0b0000001, {0b00011101, 0b00100010}},
        {0b0000010, {0b00101101, 0b00010010}},
        {0b0000011, {0b00110001, 0b00110001}},
        {0b0000100, {0b00110101, 0b00001010}},
        {0b0000101, {0b00101001, 0b00101001}},
        {0b0000110, {0b00011001, 0b00011001}},
        {0b0000111, {0b00111000, 0b00000111}},
        {0b0001000, {0b00111001, 0b00000110}},
        {0b0001001, {0b00100101, 0b00100101}},
        {0b0001010, {0b00010101, 0b00010101}},
        {0b0001011, {0b00110100, 0b00110100}},
        {0b0001100, {0b00001101, 0b00001101}},
        {0b0001101, {0b00101100, 0b00101100}},
        {0b0001110, {0b00011100, 0b00011100}},
        {0b0001111, {0b00010111, 0b00101000}},
        // 16 - 31
        {0b0010000, {0b00011011, 0b00100100}},
        {0b0010001, {0b00100011, 0b00100011}},
        {0b0010010, {0b00010011, 0b00010011}},
        {0b0010011, {0b00110010, 0b00110010}},
        {0b0010100, {0b00001011, 0b00001011}},
        {0b0010101, {0b00101010, 0b00101010}},
        {0b0010110, {0b00011010, 0b00011010}},
        {0b0010111, {0b00111010, 0b00000101}},
        {0b0011000, {0b00110011, 0b00001100}},
        {0b0011001, {0b00100110, 0b00100110}},
        {0b0011010, {0b00010110, 0b00010110}},
        {0b0011011, {0b00110110, 0b00001001}},
        {0b0011100, {0b00001110, 0b00001110}},
        {0b0011101, {0b00101110, 0b00010001}},
        {0b0011110, {0b00011110, 0b00100001}},
        {0b0011111, {0b00101011, 0b00010100}}};

    /**
     * @brief 5b/6b decoding table 
     * @details This table basically inverts the 5b/6b encoding table.
     *          The storage format is: {abcdei, EDCBA}, where abcdei is the 6b input and EDCBA is the 5b output.
     *          When decoding it is not neccessar to track running disparity, so only one value is stored.
     *          Each 5 bit / 6 bit code is extended into std::uint8_t to make it easier to use in the code.
     * 
     */
    const std::unordered_map<std::uint8_t, std::uint8_t> DECODE_5b6b = {
        {0b00100111, 0b0000000},
        {0b00011101, 0b0000001},
        {0b00101101, 0b0000010},
        {0b00110001, 0b0000011},
        {0b00110101, 0b0000100},
        {0b00101001, 0b0000101},
        {0b00011001, 0b0000110},
        {0b00111000, 0b0000111},
        {0b00111001, 0b0001000},
        {0b00100101, 0b0001001},
        {0b00010101, 0b0001010},
        {0b00110100, 0b0001011},
        {0b00001101, 0b0001100},
        {0b00101100, 0b0001101},
        {0b00011100, 0b0001110},
        {0b00010111, 0b0001111},
        {0b00011011, 0b0010000},
        {0b00100011, 0b0010001},
        {0b00010011, 0b0010010},
        {0b00110010, 0b0010011},
        {0b00001011, 0b0010100},
        {0b00101010, 0b0010101},
        {0b00011010, 0b0010110},
        {0b00111010, 0b0010111},
        {0b00110011, 0b0011000},
        {0b00100110, 0b0011001},
        {0b00010110, 0b0011010},
        {0b00110110, 0b0011011},
        {0b00001110, 0b0011100},
        {0b00101110, 0b0011101},
        {0b00011110, 0b0011110},
        {0b00101011, 0b0011111},

        {0b00011000, 0b0000000},
        {0b00100010, 0b0000001},
        {0b00010010, 0b0000010},
        {0b00110001, 0b0000011},
        {0b00001010, 0b0000100},
        {0b00101001, 0b0000101},
        {0b00011001, 0b0000110},
        {0b00000111, 0b0000111},
        {0b00000110, 0b0001000},
        {0b00100101, 0b0001001},
        {0b00010101, 0b0001010},
        {0b00110100, 0b0001011},
        {0b00001101, 0b0001100},
        {0b00101100, 0b0001101},
        {0b00011100, 0b0001110},
        {0b00101000, 0b0001111},
        {0b00100100, 0b0010000},
        {0b00100011, 0b0010001},
        {0b00010011, 0b0010010},
        {0b00110010, 0b0010011},
        {0b00001011, 0b0010100},
        {0b00101010, 0b0010101},
        {0b00011010, 0b0010110},
        {0b00000101, 0b0010111},
        {0b00001100, 0b0011000},
        {0b00100110, 0b0011001},
        {0b00010110, 0b0011010},
        {0b00001001, 0b0011011},
        {0b00001110, 0b0011100},
        {0b00010001, 0b0011101},
        {0b00100001, 0b0011110},
        {0b00010100, 0b0011111}
        };

    /**
     * @brief 3b/4b encoding table
     * @details This table is taken from https://en.wikipedia.org/wiki/8b/10b_encoding.
     *          The storage format is: {HGF, {fghj: if RD == -1, fghj: if RD == 1}},
     *          where HGF is the 3b input and fghj is the 4b output.
     *          RD again is running disparity.
     *          Each 3 bit / 4 bit code is extended into std::uint8_t to make it easier to use in the code.
     * 
     */
    const std::unordered_map<std::uint8_t, std::pair<std::uint8_t, std::uint8_t>> CODE_3b4b = {
        // 0 - 6
        {0b00000000, {0b00001011, 0b00000100}},
        {0b00000001, {0b00001001, 0b00001001}},
        {0b00000010, {0b00000101, 0b00000101}},
        {0b00000011, {0b00001100, 0b00000011}},
        {0b00000100, {0b00001101, 0b00000010}},
        {0b00000101, {0b00001010, 0b00001010}},
        {0b00000110, {0b00000110, 0b00000110}},
        // 2 variants for byte 7
        {0b00000111, {0b00001110, 0b00000001}}, // D.x.P7 - when not using D.x.A7
        {0b00000111, {0b00001000, 0b00001000}}  // D.x.A7 - used when RD == -1 and x in {17, 18, 20}
                                                //               when RD == +1 and x in {11, 13, 14}
    };

    
    /**
     * @brief 3b/4b decoding table
     * @details This table basically inverts the 3b/4b encoding table.
     *          The storage format is: {fghj, HGF}, where fghj is the 4b input and HGF is the 3b output.
     *          When decoding it is not neccessar to track running disparity, so only one value is stored.
     *          Each 3 bit / 4 bit code is extended into std::uint8_t to make it easier to use in the code.
     * 
     */
    const std::unordered_map<std::uint8_t, std::uint8_t> DECODE_3b4b = {       
        {0b00001011, 0b00000000}, 
        {0b00001001, 0b00000001}, 
        {0b00000101, 0b00000010}, 
        {0b00001100, 0b00000011}, 
        {0b00001101, 0b00000100}, 
        {0b00001010, 0b00000101}, 
        {0b00000110, 0b00000110}, 
        {0b00001110, 0b00000111},  
        {0b00001000, 0b00000111}, 
        {0b00000100, 0b00000000},
        {0b00001001, 0b00000001},
        {0b00000101, 0b00000010},
        {0b00000011, 0b00000011},
        {0b00000010, 0b00000100},
        {0b00001010, 0b00000101},
        {0b00000110, 0b00000110},
        {0b00000001, 0b00000111},
        {0b00001000, 0b00000111}
    };
};
