#include <iostream>
#include <map>

#include "Network.hpp"

#include "csv.hpp"

const std::string RESOURCES_PATH = "../resources/";
const std::string TRAIN_DATA = RESOURCES_PATH + "train.csv";
const std::string TEST_DATA  = RESOURCES_PATH + "test.csv";

constexpr float MAX_AGE = 100;
constexpr float MAX_CLASS = 3;

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
        InputParameter.Class = eachRow["Pclass"].get<float>() / MAX_CLASS;
        std::cout << eachRow["Survived"].get<float>() << " ";

        //OutputMap.insert({rowCount, eachRow["Survived"].get<float>()});
        InputParameterMap.insert({rowCount, InputParameter});
        rowCount++;
        if(rowCount == 10)
            break;
    }
    std::cout << "\n";

    Network network;
    network.Train(InputParameterMap, OutputParameterMap);

    return 0;
}