#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <unordered_map>

#include "constants.hpp"

class FileSaver {
private:
    // === SOURCE FILE ACCESS ===
    std::string sourcePath;
    std::ifstream sourceFile;

    // === TRAIN FILES ACCESS ===
    // paths
    std::string cleanTrainPath;
    std::string errorTrainPath;
    std::string xorTrainPath;
    std::string descTrainPath;
    // filestreams
    std::ofstream cleanTrainFile;
    std::ofstream errorTrainFile;
    std::ofstream xorTrainFile;
    std::ofstream descTrainFile;

    // === TEST FILES ACCESS ===
    // paths
    std::string cleanTestPath;
    std::string errorTestPath;
    std::string xorTestPath;
    std::string descTestPath;
    // filestreams
    std::ofstream cleanTestFile;
    std::ofstream errorTestFile;
    std::ofstream xorTestFile;
    std::ofstream descTestFile;


public:
    /**
     * @brief Construct a new File Saver object
     * 
     * @param _sourcePath     path to the source file
     * @param _cleanTrainPath path to the clean train file
     * @param _errorTrainPath path to the error train file
     * @param _xorTrainPath   path to the xor train file
     * @param _descTrainPath  path to the desc train file
     * @param _cleanTestPath  path to the clean test file
     * @param _errorTestPath  path to the error test file
     * @param _xorTestPath    path to the xor test file
     * @param _descTestPath   path to the desc test file
     */
    FileSaver(const std::string& _sourcePath,
              const std::string& _cleanTrainPath,
              const std::string& _errorTrainPath,
              const std::string& _xorTrainPath,
              const std::string& _descTrainPath,
              const std::string& _cleanTestPath,
              const std::string& _errorTestPath,
              const std::string& _xorTestPath,
              const std::string& _descTestPath);

    /**
     * @brief Construct a new File Saver object, with default directory structure
     * @details Default directory structure (with sourceFileName = source.txt, exitFileName = exit):
     *         - data/
     *           - raw/
     *               - source.txt
     *           - prep/
     *               - train/
     *                   - exit_og.dat          clean train data
     *                   - exit.dat             error train data       
     *                   - exit_xor.dat         xor train data
     *                   - exit_errDesc.csv     error description train data
     *               - test/
     *                   - exit_og.dat          clean train data
     *                   - exit.dat             error train data       
     *                   - exit_xor.dat         xor train data
     *                   - exit_errDesc.csv     error description train data
     *
     * 
     * @param dataDirPath    path to the data/ directory, ex. ../data/
     * @param sourceFileName name of the source file (with extension), ex. source.txt
     * @param exitFileName   basic name of the exit files (without extension), ex. exit
     */
    FileSaver(const std::string& dataDirPath,
              const std::string& sourceFileName,
              const std::string& exitFileName);

    ~FileSaver();

    /**
     * @brief Opens all the filestreams, returns false if any of them failed to open
     * 
     * @return true all filestreams opened successfully
     * @return false any of the filestreams failed to open, error message in cerr
     */
    bool openFiles();

    /**
     * @brief Closes all the filestreams
     * 
     */
    void closeFiles();

    /**
     * @brief Writes given data to appropriate files, returns false if any of the writes failed
     * 
     * @param cleanFrame frame without errors, as bytesVec
     * @param errorFrame frame with errors present, as bytesVec
     * @param xorFrame XOR of cleanFrame and errorFrame, as bytesVec
     * @param desc map of the positions of errors within frames
     * @param isTrain should the data be written to train files or test files
     * @return true all writes successful
     * @return false any of the writes failed, error message in cerr
     */
    bool write(const bytesVec& cleanFrame, 
               const bytesVec& errorFrame, 
               const bytesVec& xorFrame, 
               const std::unordered_map<std::string, int>& desc,
               bool isTrain);

    std::ifstream& source() { return sourceFile; }

};