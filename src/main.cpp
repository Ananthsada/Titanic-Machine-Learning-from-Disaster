#include <iostream>
#include <map>

#include "Network.hpp"

#include "csv.hpp"
#include "Vengai.hpp"

const std::string RESOURCES_PATH = "../resources/";
const std::string TRAIN_DATA = RESOURCES_PATH + "train.csv";
const std::string TEST_DATA  = RESOURCES_PATH + "test.csv";

constexpr float MAX_AGE = 100;
constexpr float MAX_CLASS = 3;

#if 0
int main()
{
    csv::CSVReader Reader(TRAIN_DATA);

    InputParameterMapType InputParameterMap;
    OutputParameterMapType OutputParameterMap;
    uint16_t rowCount = 0;
    for(auto& eachRow : Reader)
    {
        InputParameterStruct InputParameter;
        if(!eachRow["Age"].is_null())
        {
            InputParameter.Age = eachRow["Age"].get<float>() / MAX_AGE;
        }
        else
        {
            InputParameter.Age = 1.0f / MAX_AGE;
        }
        InputParameter.Class = eachRow["Pclass"].get<float>() / MAX_CLASS;

        std::string Gender = eachRow["Sex"].get<std::string>();

        if(Gender == "male")
        {
            InputParameter.Gender = 1.0f;
        }
        else
        {
            InputParameter.Gender = 0.0f;
        }

        float out = eachRow["Survived"].get<float>();
        //std::cout << out << " ";

        //OutputMap.insert({rowCount, eachRow["Survived"].get<float>()});
        InputParameterMap.insert({rowCount, InputParameter});
        OutputParameterMap.insert({rowCount, out});
        rowCount++;
        if(rowCount == 10)
            break;
    }
    std::cout << "\n";

    Network network;
    network.Train(InputParameterMap, OutputParameterMap);

    csv::CSVReader TestReader(TEST_DATA);

    rowCount = 0;
    int sum = 0;
    std::ofstream ss("./output.csv");
    auto writer = csv::make_csv_writer(ss);
    writer << std::vector<std::string>({"PassengerId", "Survived"});

    for(auto& eachRow : TestReader)
    {
        InputParameterStruct InputParameter;
        if(!eachRow["Age"].is_null())
        {
            InputParameter.Age = eachRow["Age"].get<float>() / MAX_AGE;
        }
        else
        {
            InputParameter.Age = 1.0f / MAX_AGE;
        }
        InputParameter.Class = eachRow["Pclass"].get<float>() / MAX_CLASS;
        //float out = eachRow["Survived"].get<float>();


        std::string Gender = eachRow["Pclass"].get<std::string>();

        if(Gender == "male")
        {
            InputParameter.Gender = 1.0f;
        }
        else
        {
            InputParameter.Gender = 0.0f;
        }

        float output = network.getOutput(InputParameter);
        //std::cout << "Actual:" << out << " From NN:" << output << "\n";

        // if(fabs(out - output) != 0.0f)
        // {
        //     //std::cout << "Non-Zero\n";
        //     sum++;
        // }
        // else
        // {
        //     //std::cout << "Zero\n";
        // }

        int passengerId = eachRow["PassengerId"].get<int>();

        writer << std::vector<int>({passengerId, static_cast<int>(output)});

        rowCount++;
        if(rowCount == 10)
            break;
    }
    float percentage = sum / (float)rowCount;

    std::cout << "Percentage:" << percentage << "\n";

    ss.close();

    return 0;
}

#else
int main()
{
    Vengai::NetworkConfig networkConfig;
    networkConfig.mInputCount = 3;
    networkConfig.mLayerCount = 2;
    networkConfig.mOutputCount = 1;
    networkConfig.mNodeCount.emplace_back(2);
    networkConfig.mNodeCount.emplace_back(1);

    Vengai::Network network(networkConfig);
    csv::CSVReader Reader(TRAIN_DATA);

    InputParameterMapType InputParameterMap;
    OutputParameterMapType OutputParameterMap;
    uint16_t rowCount = 0;
    Vengai::NetworkInputType NetworkInput;
    Vengai::NetworkOutputType NetworkOutput;
    for(auto& eachRow : Reader)
    {
        std::vector<float> eachInput;
        if(!eachRow["Age"].is_null())
        {
            eachInput.emplace_back(eachRow["Age"].get<float>() / MAX_AGE);
        }
        else
        {
            eachInput.emplace_back(1.0f / MAX_AGE);
        }
        eachInput.emplace_back(eachRow["Pclass"].get<float>() / MAX_CLASS);

        std::string Gender = eachRow["Sex"].get<std::string>();

        if(Gender == "male")
        {
            eachInput.emplace_back(1.0f);
        }
        else
        {
            eachInput.emplace_back(0.0f);
        }

        NetworkOutput.emplace_back(eachRow["Survived"].get<float>());
        
        NetworkInput.emplace_back(eachInput);

        rowCount++;
        if(rowCount == 10)
            break;
    }

    network.train(NetworkInput, NetworkOutput);

    return 0;

    csv::CSVReader TestReader(TEST_DATA);

    rowCount = 0;
    int sum = 0;
    std::ofstream ss("./output.csv");
    auto writer = csv::make_csv_writer(ss);
    writer << std::vector<std::string>({"PassengerId", "Survived"});

    NetworkInput.clear();
    for(auto& eachRow : TestReader)
    {
        std::vector<float> eachInput;
        if(!eachRow["Age"].is_null())
        {
            eachInput.emplace_back(eachRow["Age"].get<float>() / MAX_AGE);
        }
        else
        {
            eachInput.emplace_back(1.0f / MAX_AGE);
        }
        eachInput.emplace_back(eachRow["Pclass"].get<float>() / MAX_CLASS);
        //float out = eachRow["Survived"].get<float>();


        std::string Gender = eachRow["Pclass"].get<std::string>();

        if(Gender == "male")
        {
            eachInput.emplace_back(1.0f);
        }
        else
        {
            eachInput.emplace_back(0.0f);
        }

        float output = network.test(eachInput);

        int passengerId = eachRow["PassengerId"].get<int>();

        writer << std::vector<int>({passengerId, static_cast<int>(output)});

        rowCount++;
        // if(rowCount == 50)
        //     break;
    }
    float percentage = sum / (float)rowCount;

    std::cout << "Percentage:" << percentage << "\n";

    ss.close();

    return 0;
}
#endif
