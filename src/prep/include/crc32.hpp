// Author: Marek Szymański
// Description: CRC32 reference table and calculation from vector of uint8_t

#pragma once

#include <cstdint>
#include <vector>

#include "constants.hpp"

using byte = std::uint8_t;
using bytesVec = std::vector<byte>;

namespace crc32 {
    /**
     * @brief Generates CRC32 reference table, credit to https://github.com/timepp
     * @details Generates a reference table of 256 CRC32 values, used for CRC32 calculation
     *          Implementation by https://gist.github.com/timepp/1f678e200d9e0f2a043a9ec6b3690635
     *          Only changed type of j to int from size_t
     * 
     * @param table pointer to the table to be filed with CRC32 values
     */
	void generate_table(uint32_t *table);

    /**
     * @brief Calculates CRC32 of the given bytesVec, using pregenerated CRC32 reference table
     * 
     * @param table     pointer to the pregenerated CRC32 reference table (genreted by generate_table())
     * @param data      vector of bytes to calculate CRC32 of
     * @return uint32_t CRC32 value of the data, as uint32_t 
     */
	uint32_t crc(uint32_t* table, const bytesVec& data);

    /**
     * @brief Converts CRC32 value from uint32_t to bytesVec of 4 bytes
     * 
     * @param crc       CRC32 value to be converted
     * @return bytesVec vector of 4 bytes equal to the CRC32 value of crc 
     */
    bytesVec toBytesVec(uint32_t crc);
};