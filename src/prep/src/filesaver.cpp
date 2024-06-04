#include "../include/filesaver.hpp"

bool FileSaver::openFiles() {
    // open source file
    sourceFile.open(sourcePath);
    if (!sourceFile.is_open()) {
        std::cerr << "Failed to open source file: " << sourcePath << std::endl;
        return false;
    }

    // open train files
    cleanTrainFile.open(cleanTrainPath);
    if (!cleanTrainFile.is_open()) {
        std::cerr << "Failed to open clean train file: " << cleanTrainPath << std::endl;
        return false;
    }

    errorTrainFile.open(errorTrainPath);
    if (!errorTrainFile.is_open()) {
        std::cerr << "Failed to open error train file: " << errorTrainPath << std::endl;
        return false;
    }

    xorTrainFile.open(xorTrainPath);
    if (!xorTrainFile.is_open()) {
        std::cerr << "Failed to open xor train file: " << xorTrainPath << std::endl;
        return false;
    }

    descTrainFile.open(descTrainPath);
    if (!descTrainFile.is_open()) {
        std::cerr << "Failed to open desc train file: " << descTrainPath << std::endl;
        return false;
    }

    // open test files
    cleanTestFile.open(cleanTestPath);
    if (!cleanTestFile.is_open()) {
        std::cerr << "Failed to open clean test file: " << cleanTestPath << std::endl;
        return false;
    }

    errorTestFile.open(errorTestPath);
    if (!errorTestFile.is_open()) {
        std::cerr << "Failed to open error test file: " << errorTestPath << std::endl;
        return false;
    }

    xorTestFile.open(xorTestPath);
    if (!xorTestFile.is_open()) {
        std::cerr << "Failed to open xor test file: " << xorTestPath << std::endl;
        return false;
    }

    descTestFile.open(descTestPath);
    if (!descTestFile.is_open()) {
        std::cerr << "Failed to open desc test file: " << descTestPath << std::endl;
        return false;
    }

    return true;
}

void FileSaver::closeFiles() {
    sourceFile.close();
    cleanTrainFile.close();
    errorTrainFile.close();
    xorTrainFile.close();
    descTrainFile.close();
    cleanTestFile.close();
    errorTestFile.close();
    xorTestFile.close();
    descTestFile.close();
}

FileSaver::~FileSaver() {
    closeFiles();
}

bool FileSaver::write(const bytesVec& cleanFrame, 
               const bytesVec& errorFrame, 
               const bytesVec& xorFrame, 
               const std::unordered_map<std::string, int>& desc,
               bool isTrain) {
    if (isTrain) {
        // write to train files

        // write clean to file
        cleanTrainFile.write((char*)cleanFrame.data(), cleanFrame.size());
        if (cleanTrainFile.fail()) {
            std::cerr << "Failed to write to clean train file" << std::endl;
            return false;
        }

        // write error to file
        errorTrainFile.write((char*)errorFrame.data(), errorFrame.size());
        if (errorTrainFile.fail()) {
            std::cerr << "Failed to write to error train file" << std::endl;
            return false;
        }

        // write xor to file
        xorTrainFile.write((char*)xorFrame.data(), xorFrame.size());
        if (xorTrainFile.fail()) {
            std::cerr << "Failed to write to xor train file" << std::endl;
            return false;
        }

        // write desc to file
        for (auto it = desc.begin(); it != desc.end(); it++)
        {
            if (std::next(it) == desc.end())
            {
                descTrainFile << it->second << "\n";
            }
            else
            {
                descTrainFile << it->second << ",";
            }
        }
        if (descTrainFile.fail()) {
            std::cerr << "Failed to write to desc train file" << std::endl;
            return false;
        }

        return true;
    } else {
        // write to test files

        // write clean to file
        cleanTestFile.write((char*)cleanFrame.data(), cleanFrame.size());
        if (cleanTestFile.fail()) {
            std::cerr << "Failed to write to clean train file" << std::endl;
            return false;
        }

        // write error to file
        errorTestFile.write((char*)errorFrame.data(), errorFrame.size());
        if (errorTestFile.fail()) {
            std::cerr << "Failed to write to error train file" << std::endl;
            return false;
        }

        // write xor to file
        xorTestFile.write((char*)xorFrame.data(), xorFrame.size());
        if (xorTestFile.fail()) {
            std::cerr << "Failed to write to xor train file" << std::endl;
            return false;
        }

        // write desc to file
        for (auto it = desc.begin(); it != desc.end(); it++)
        {
            if (std::next(it) == desc.end())
            {
                descTestFile << it->second << "\n";
            }
            else
            {
                descTestFile << it->second << ",";
            }
        }
        if (descTestFile.fail()) {
            std::cerr << "Failed to write to desc train file" << std::endl;
            return false;
        }

        return true;
    }

}

FileSaver::FileSaver(const std::string& _sourcePath,
              const std::string& _cleanTrainPath,
              const std::string& _errorTrainPath,
              const std::string& _xorTrainPath,
              const std::string& _descTrainPath,
              const std::string& _cleanTestPath,
              const std::string& _errorTestPath,
              const std::string& _xorTestPath,
              const std::string& _descTestPath) {
    sourcePath = _sourcePath;

    cleanTrainPath = _cleanTrainPath;
    errorTrainPath = _errorTrainPath;
    xorTrainPath = _xorTrainPath;
    descTrainPath = _descTrainPath;

    cleanTestPath = _cleanTestPath;
    errorTestPath = _errorTestPath;
    xorTestPath = _xorTestPath;
    descTestPath = _descTestPath;
}

FileSaver::FileSaver(const std::string& dataDirPath,
              const std::string& sourceFileName,
              const std::string& exitFileName) {
    sourcePath = dataDirPath + "raw/" + sourceFileName;

    cleanTrainPath = dataDirPath + "prep/train/" + sourceFileName + "_og.dat";
    errorTrainPath = dataDirPath + "prep/train/" + sourceFileName + ".dat";
    xorTrainPath = dataDirPath + "prep/train/" + sourceFileName + "_xor.dat";
    descTrainPath = dataDirPath + "prep/train/" + sourceFileName + "_errDesc.csv";

    cleanTestPath = dataDirPath + "prep/test/" + sourceFileName + "_og.dat";
    errorTestPath = dataDirPath + "prep/test/" + sourceFileName + ".dat";
    xorTestPath = dataDirPath + "prep/test/" + sourceFileName + "_xor.dat";
    descTestPath = dataDirPath + "prep/test/" + sourceFileName + "_errDesc.csv";
}