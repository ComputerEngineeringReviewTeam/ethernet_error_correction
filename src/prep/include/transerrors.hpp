// Author: Marek Szymański
// Description: Flipping bits in the fields of a 8b/10b encoded Ethernet II frame

#pragma once

#include <vector>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <random>

#include "constants.hpp"

using byte = std::uint8_t;
using bytesVec = std::vector<byte>;
using symbol10 = std::uint16_t;

namespace transerrors {

    /**
    * @brief Negates bits at given positions in given vector of bytes
    * @details If the positions would be {0, 12, 17} then bits negated would be: 
    *          0th bit of 0th byte, 4th bit of 1st byte, 1st bit of 3rd byte and so on.
    *          Positions out of range are ignored.    
    * 
    * @param data      vector of original bytes, not modified
    * @param positions vector of positions to negate bits at, should be no longer than 8*data.size() (nr of bits in data)
    * @return          vector of bytes with bits negated at given positions
    */
    bytesVec flipBits(const bytesVec&data, const std::unordered_set<int>& positions);

    /**
     * @brief Get the position of random bit in destination MAC field of 8b/10b encoded Ethernet II frame
     * @details Returns random position of single bit within the 'destination MAC' field of a 8b/10b encoded Ethernet II frame
     *          the 'destination MAC' field is the first 6 bytes of the frame,
     *          which after encoding are the first 60 bits of the encoded frame (bytes 0 - 7 and the older half of byte 8)      
     * 
     * @param gen  mt19937 random number generator
     * @return int position of the bit
     */
    int getPositionsInFirstEncodedMAC(std::mt19937& gen);

    /**
     * @brief Get the position of random bit in source MAC field of 8b/10b encoded Ethernet II frame
     * @details Returns random position of single bit within the 'source MAC' field of a 8b/10b encoded Ethernet II frame
     *          the 'source MAC' field are the bytes 7 - 12 of the frame,
     *          which after encoding are the bits 61 - 119 (indexing from 0) of the encoded frame (younger half of byte 8 and bytes 9 - 14)      
     * 
     * @param gen  mt19937 random number generator
     * @return int position of the bit
     */
    int getPositionsInSecondEncodedMAC(std::mt19937& gen);

    /**
     * @brief Get the position of random bit in EtherType field of 8b/10b encoded Ethernet II frame
     * @details Returns random position of single bit within the 'EtherType' field of a 8b/10b encoded Ethernet II frame
     *          the 'EtherTypes' field are the bytes 13 - 14 of the frame,
     *          which after encoding are the bits 120 - 139 (indexing from 0) of the encoded frame (bytes 15 - 16 and the older half of byte 17)      
     * 
     * @param gen  mt19937 random number generator
     * @return int position of the bit
     */
    int getPositionsInEncodedEtherType(std::mt19937& gen);

    /**
     * @brief Get the position of random bit in CRC field of 8b/10b encoded Ethernet II frame
     * @details Returns random position of single bit within the 'CRC' field of a 8b/10b encoded Ethernet II frame
     *          the 'CRC' field are the last 4 bytes (bytes size-4 - size-1) of the frame,
     *          which after encoding are the last 40 bits of the encoded frame 
     *          as such the length of the frame before encoding (which might introduce padding) should be passed as an argument
     * 
     * @param gen            mt19937 random number generator
     * @param plainframeSize length of the in bytes frame before encoding (which might introduce padding)
     * @return int           position of the bit
     */
    int getPositionsInEncodedCRC(std::mt19937& gen, int plainframeSize);

    /**
     * @brief Get the position of random bit in Data field of 8b/10b encoded Ethernet II frame
     * @details Returns random position of single bit within the 'Data' field of a 8b/10b encoded Ethernet II frame
     *          the 'Data' field are the bytes 15 - size-5 of the frame (all bytes between the header and the last 4 bytes of CRC)
     *          which after encoding are the bits 140 - size*8 - 40 (indexing from 0) of the encoded frame 
     *          as such the length of the frame before encoding (which might introduce padding) should be passed as an argument
     * 
     * @param gen            mt19937 random number generator
     * @param plainframeSize length of the frame in bytes before encoding (which might introduce padding)
     * @return int           position of the bit
     */
    int getPositionsInEncodedFrameData(std::mt19937& gen, int plainframeSize);

    /**
     * @brief Get the positions of random  bits within the fields of a 8b/10b encoded Ethernet II frame according to given map
     * @details Returns random positions of single bits within the given fields of a 8b/10b encoded Ethernet II frame   
     *          The map details how many errors should be introduced in each field
     *          The format would be: {{"DestMAC", (0-60)}, {"SourceMAC", (0-60)}, {"EtherType", (0-20)}, {"CRC", (0-*)}, {"Data", (0-40)}}
     *          possible ranges for each field given in the brackets
     *          (* - length of the frame in bytes before encoding * 10 - 140 (Ethernet II header) - 40 (CRC) ).
     *          Fields not present in the map are ignored
     *          The function returns std::unordered_set of unique positions of bits to be flipped (to be used in flipBits() function or inspected)
     * 
     * 
     * @param targets                  map of fields and number of errors to introduce in each field
     * @param gen                      mt19937 random number generator
     * @param plainframeSize           length of the frame before encoding (which might introduce padding)
     * @return std::unordered_set<int> set of positions of bits to be flipped
     */
    std::unordered_set<int> getPositionsInEncodedFrame(const std::unordered_map<std::string, int>& targets, std::mt19937& gen, int plainframeSize);

    /**
     * @brief Flips given number of randomly chosen bits in the given vector of bytes
     * 
     * @param data      vector of bytes to be modified
     * @param count     number of bits to flip
     * @param gen       mt19937 random number generator
     * @return bytesVec new vector of bytes with flipped bits
     */
    bytesVec flipRandomBits(const bytesVec& data, int count, std::mt19937& gen);

    /**
     * @brief Flips at least 1 bit in each byte of the given vector of bytes, more bits might be flipped according to the given probabilities
     * 
     * @param data      vector of bytes to be modified
     * @param chances   vector of probabilities of flipping bits in each byte, each should be in range [0, 1], 
     *                  sizes longer than data will not result in more than data.size() bits flipped
     * @param gen       mt19937 random number generator
     * @return bytesVec new vector of bytes with flipped bits
     */
    bytesVec flipRandomBits(const bytesVec& data, const std::vector<double>& chances, std::mt19937& gen);
};