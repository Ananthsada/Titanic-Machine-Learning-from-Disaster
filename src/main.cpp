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

        std::string Gender = eachRow["Pclass"].get<std::string>();

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
        // if(rowCount == 10)
        //     break;
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
        // if(rowCount == 50)
        //     break;
    }
    float percentage = sum / (float)rowCount;

    std::cout << "Percentage:" << percentage << "\n";

    ss.close();

    return 0;
}