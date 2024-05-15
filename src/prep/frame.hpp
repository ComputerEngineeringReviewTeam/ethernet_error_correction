// Author: Marek Szymański
// Description: Operations on Ethernet II frames, reading from file, parsing, padding, printing, writing to binary file

#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include <sstream>
#include <cstdint>

using byte = std::uint8_t;
using bytesVec = std::vector<byte>;
using symbol10 = std::uint16_t;

const int BYTE_SIZE = 8;
const int BYTE_MAX = 256;

const int ETH2_HEADER_SIZE = 14;
const int MAC_SIZE = 6;
const int ETH2_SRC_MAC_OFFSET = 6;
const int ETH2_DST_MAC_OFFSET = 0;
const int ETH2_TYPE_SIZE = 2;
const int ETH2_TYPE_OFFSET = 12;
const int ETH2_PAYLOAD_OFFSET = 14;
const int ETH2_MIN_PAYLOAD_SIZE = 46;
const int ETH2_MAX_PAYLOAD_SIZE = 1500;
const int ETH2_MIN_FRAME_SIZE = 64;
const int ETH2_MAX_FRAME_SIZE = 1518;
const int ETH2_CRC_SIZE = 4;
const int ETH2_CRC_BACK_OFFSET = 4;

const int IP4_HEADER_SIZE = 20;
const int IP4_ADDR_SIZE = 4;
const int IP4_SRC_ADDR_OFFSET = 12;
const int IP4_DST_ADDR_OFFSET = 16;

/**
 * @brief Creates bytesVec with 6 random bytes, immulating MAC address
 * 
 * @return bytesVec 
 */
bytesVec randomMAC();

/**
 * @brief Parses frame from string to bytesVec
 * @details Frame is in the format of Wireshark capture, ie.
 *          chain of bytes written in hexadecimal, separated by '|'
 * 
 * @param line      string representation a Ethernet frame
 * @return bytesVec vector of bytes
 */
bytesVec parseFrame(std::string line);

/**
 * @brief Parses frames from file to vector of bytesVec
 * @details Opens the Wireshark capture file pointed to by filename and reads frames from it
 *          The file contains Ethernet frames written in the following format: |0   |32|4a|1c|09|f2|55|7d|9a|...
 *          As additional informations are added by Wireshark (timestamp etc.), the function filters only lines starting with '|0'
 *          To make parsing easier the beginning "|0   " is removed from each line containing a frame
 * 
 * @param filename               name of the file containing frames
 * @return std::vector<bytesVec> vector of frames, each frame is a vector of bytes
 */
std::vector<bytesVec> readFrames(std::string filename);

/**
 * @brief Prints bytesVec with std::cout as series of bytes in hex, separated by ' ' and followed by '::' and size in bytes
 * 
 * @param bytes vector of bytes to be printed
 */
void printBytesVec(const bytesVec& bytes);

/**
 * @brief Write bytesVec as vector bytes to binary file
 * 
 * @param bytes    vector of bytes to be written
 * @param filename name of the file to write to
 */
void writeBytesVecToBinary(const bytesVec& bytes, std::string filename);

/**
 * @brief Padds the bytesVec with given (or default = 0) padding to the given size
 * 
 * @param bytes     vector of bytes to be padded, not modified
 * @param size      size to which the vector should be padded
 * @param padding   byte to be used as padding, default is 0
 * @return bytesVec padded vector of bytes
 */
bytesVec rightPadBytesVec(const bytesVec& bytes, int size, byte padding = 0);

/**
 * @brief Replaces the MAC address in the given Ethernet II frame with a random ones
 * 
 * @param frame     Ethernet II frame as vector of bytes
 * @param mode      0 - randomize only destination MAC address
 *                  1 - randomize only source MAC address
 *                  2 - randomize both source and destination MAC addresses
 * @return bytesVec frame with randomized MAC addresses
 */
bytesVec randomizeMAC(const bytesVec& frame, int mode);

// IPv4

/**
 * @brief Replaces the IPv4 address in the given IPv4 packet with a random ones
 * 
 * @param frame     Ethernet II frame, containing IPv4 packet, as vector of bytes
 * @param mode      0 - randomize only source IPv4 address
 *                  1 - randomize only destination IPv4 address
 *                  2 - randomize both source and destination IPv4 addresses
 * @return bytesVec Ethernet II frame with randomized IPv4 addresses
 */
bytesVec randomizeIPv4Addr(const bytesVec& frame, int mode);