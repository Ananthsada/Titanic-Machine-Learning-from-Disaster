#include <iostream>
#include <map>

#include "Network.hpp"

#include "csv.hpp"

const std::string RESOURCES_PATH = "../resources/";
const std::string TRAIN_DATA = RESOURCES_PATH + "train.csv";
const std::string TEST_DATA  = RESOURCES_PATH + "test.csv";

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
            InputParameter.Age = eachRow["Age"].get<float>();
        }
        InputParameter.Class = eachRow["Pclass"].get<float>();

        //OutputMap.insert({rowCount, eachRow["Survived"].get<float>()});
        InputParameterMap.insert({rowCount, InputParameter});
        rowCount++;
    }

    Network network;
    network.Train(InputParameterMap, OutputParameterMap);

    return 0;
}