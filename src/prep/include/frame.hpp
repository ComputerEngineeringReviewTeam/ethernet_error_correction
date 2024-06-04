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

#include "constants.hpp"


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

/**
 * @brief Reads frames from binary file
 * 
 * @param filename               name of the file containing frames
 * @param frameSize              size of each frame in bytes
 * @return std::vector<bytesVec> vector of frames, each frame is a vector of bytes
 */
std::vector<bytesVec> readFramesFromBinary(std::string filename, int frameSize);

/**
 * @brief Get the Ether Type of the Ethernet II frame (given as bytesVec) as std::uint32_t
 * 
 * @param frame Ethernet II frame as vector of bytes
 * @return std::uint32_t EtherType of the Ethernet II frame
 */
std::uint32_t getEtherType(const bytesVec& frame);

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

