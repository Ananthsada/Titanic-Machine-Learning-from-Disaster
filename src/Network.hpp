#ifndef __NETWORK_HPP__
#define __NETWORK_HPP__

#include <iostream>
#include <map>
#include <vector>
#include <cmath>
#include <chrono>

struct InputParameterStruct
{
    InputParameterStruct()
    {
        Age = 0.0f;
        Class = 0.0f;
    }
    float Age;
    float Class;
    float Gender;
};

using InputParameterMapType = std::map<uint16_t, InputParameterStruct>;
using OutputParameterMapType = std::map<uint16_t, float>;

constexpr uint8_t LAYER_COUNT = 1;
constexpr uint8_t HLAYER_NODE_COUNT = 2;
constexpr float SCALE_FACTOR = 1.5f;

float Sigmoid(float input)
{
    return 1 / (1 + exp(-input));
}

float SigmoidTransient(float input)
{
    float sigmoid = Sigmoid(input);

    return sigmoid * ( 1 - sigmoid);
}

class Network
{
public:
    Network();
    ~Network() {}

    void Train(const InputParameterMapType& InputParameterMap, const OutputParameterMapType& OutputParamterMap);
    float getOutput(const InputParameterStruct& InputParams);
private:
    using NodeInputParamterType = std::vector<float>;
    using NodeWeightParamterType = std::vector<float>;
    using NodeBiasParamterType = std::vector<float>;

    float mHLayerParameters[HLAYER_NODE_COUNT][4];
    float mOutputLayerParamter[3];
    OutputParameterMapType mOutputParamter;


    void printWeights();
    float NodeOuput(const NodeInputParamterType& NodeInputParameter, const NodeWeightParamterType& NodeWeightParamter);
    std::vector<float> ForwardPropogation(const InputParameterStruct& InputParams);
};

Network::Network()
{
    mHLayerParameters[0][0] = 1.0f;
    mHLayerParameters[0][1] = 0.0f;
    mHLayerParameters[0][2] = 1.0f;
    mHLayerParameters[0][3] = 1.0f;
    mHLayerParameters[1][0] = 1.0f;
    mHLayerParameters[1][1] = 0.0f;
    mHLayerParameters[1][2] = 1.0f;
    mHLayerParameters[1][3] = 1.0f;

    mOutputLayerParamter[0] = 1.0f;
    mOutputLayerParamter[1] = 0.0f;
    mOutputLayerParamter[2] = 1.0f;
}

void Network::Train(const InputParameterMapType& InputParameter, const OutputParameterMapType& OutputParamter)
{
    auto start = std::chrono::steady_clock::now();
    std::cout << "Input Paramter Size:" << InputParameter.size() << "\n";

    float PrevCost = 0.0f;
    float CurrentCost = 0.0f;
    while(true)
    {
        int index = 0;
        CurrentCost = 0.0f;
        float HLayerParamters[2][3] = {0.0f};
        float OutputLayerParamters[3] = {0.0f};
        for(const auto& each : InputParameter)
        {
            std::vector<float> _output = ForwardPropogation(each.second);

            CurrentCost += (OutputParamter.at(index) - _output[2]) * (OutputParamter.at(index) - _output[2]);
            index++;

            float h1 = _output[0];
            float h2 = _output[1];
            float o1 = _output[2];
            float delta = SigmoidTransient(mOutputLayerParamter[0] + (mOutputLayerParamter[1] * h1) + (mOutputLayerParamter[2] * h2));
            float deltah1 = delta * OutputLayerParamters[1] * SigmoidTransient(HLayerParamters[0][0] + 
                    (HLayerParamters[0][1] * each.second.Age) + (HLayerParamters[0][2] * each.second.Class) + (HLayerParamters[0][3] * each.second.Gender));
            float deltah2 = delta * OutputLayerParamters[2] * SigmoidTransient(HLayerParamters[1][0] + 
                    (HLayerParamters[1][1] * each.second.Age) + (HLayerParamters[1][2] * each.second.Class) + (HLayerParamters[1][3] * each.second.Gender));
            HLayerParamters[0][0] += deltah1;
            HLayerParamters[0][1] += deltah1 * each.second.Age;
            HLayerParamters[0][2] += deltah1 * each.second.Class;
            HLayerParamters[0][3] += deltah1 * each.second.Gender;
            HLayerParamters[1][0] += deltah2;
            HLayerParamters[1][1] += deltah2 * each.second.Age;
            HLayerParamters[1][2] += deltah2 * each.second.Class;
            HLayerParamters[1][3] += deltah2 * each.second.Gender;

            OutputLayerParamters[0] += delta;
            OutputLayerParamters[1] += delta * h1;
            OutputLayerParamters[2] += delta * h2;
            //std::cout << _output[2] << " ";
        }
        //std::cout << "\n";

        CurrentCost = CurrentCost / ( 2 * InputParameter.size());

        std::cout << "Prev:" << PrevCost << " Current:" << CurrentCost << "\n";
        if(fabs(PrevCost - CurrentCost) < 0.0001f)
        {
            break;
        }

        mHLayerParameters[0][0] = mHLayerParameters[0][0] - (SCALE_FACTOR * (HLayerParamters[0][0] / InputParameter.size()));
        mHLayerParameters[0][1] = mHLayerParameters[0][1] - (SCALE_FACTOR * (HLayerParamters[0][1] / InputParameter.size()));
        mHLayerParameters[0][2] = mHLayerParameters[0][2] - (SCALE_FACTOR * (HLayerParamters[0][2] / InputParameter.size()));
        mHLayerParameters[1][0] = mHLayerParameters[1][0] - (SCALE_FACTOR * (HLayerParamters[1][0] / InputParameter.size()));
        mHLayerParameters[1][1] = mHLayerParameters[1][1] - (SCALE_FACTOR * (HLayerParamters[1][1] / InputParameter.size()));
        mHLayerParameters[1][2] = mHLayerParameters[1][2] - (SCALE_FACTOR * (HLayerParamters[1][2] / InputParameter.size()));

        mOutputLayerParamter[0] = mOutputLayerParamter[0] - (SCALE_FACTOR * (OutputLayerParamters[0] / InputParameter.size()));
        mOutputLayerParamter[1] = mOutputLayerParamter[1] - (SCALE_FACTOR * (OutputLayerParamters[1] / InputParameter.size()));
        mOutputLayerParamter[2] = mOutputLayerParamter[2] - (SCALE_FACTOR * (OutputLayerParamters[2] / InputParameter.size()));
        
        PrevCost = CurrentCost;
    }

    float elaspedTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    std::cout << "Elapsed(us)=" << elaspedTime << std::endl;

    printWeights();
}


void Network::printWeights()
{
    std::cout << "L11:{" << mHLayerParameters[0][0] << "," << mHLayerParameters[0][1] << "," << mHLayerParameters[0][2] << "}\n";
    std::cout << "L12:{" << mHLayerParameters[1][0] << "," << mHLayerParameters[1][1] << "," << mHLayerParameters[1][2] << "}\n";
    std::cout << "L21:{" << mOutputLayerParamter[0] << "," << mOutputLayerParamter[1] << "," << mOutputLayerParamter[2] << "}\n";
}

std::vector<float> Network::ForwardPropogation(const InputParameterStruct& InputParams)
{
    /* hidden layer */
    //std::cout << "Age:" << InputParams.Age << " Class:" << InputParams.Class;
    NodeInputParamterType NodeInputParamter;
    NodeInputParamter.emplace_back(1.0f);
    NodeInputParamter.emplace_back(InputParams.Age);
    NodeInputParamter.emplace_back(InputParams.Class);
    NodeInputParamter.emplace_back(InputParams.Gender);

    NodeWeightParamterType NodeWeightParamter;
    NodeWeightParamter.emplace_back(mHLayerParameters[0][0]);
    NodeWeightParamter.emplace_back(mHLayerParameters[0][1]);
    NodeWeightParamter.emplace_back(mHLayerParameters[0][2]);
    NodeWeightParamter.emplace_back(mHLayerParameters[0][3]);
    float h1Node1Output = NodeOuput(NodeInputParamter, NodeWeightParamter);

    NodeWeightParamter.clear();
    NodeWeightParamter.emplace_back(mHLayerParameters[1][0]);
    NodeWeightParamter.emplace_back(mHLayerParameters[1][1]);
    NodeWeightParamter.emplace_back(mHLayerParameters[1][2]);
    NodeWeightParamter.emplace_back(mHLayerParameters[1][3]);
    float h1Node2Output = NodeOuput(NodeInputParamter, NodeWeightParamter);

    /* output layer */
    NodeInputParamter.clear();
    NodeInputParamter.emplace_back(1.0f);
    NodeInputParamter.emplace_back(h1Node1Output);
    NodeInputParamter.emplace_back(h1Node2Output);

    NodeWeightParamter.clear();
    NodeWeightParamter.emplace_back(mOutputLayerParamter[0]);
    NodeWeightParamter.emplace_back(mOutputLayerParamter[1]);
    NodeWeightParamter.emplace_back(mOutputLayerParamter[2]);
    
    float output = NodeOuput(NodeInputParamter, NodeWeightParamter);

    //std::cout << " Output:" << output << "\n";
    return {h1Node1Output, h1Node2Output, output};
}

float Network::NodeOuput(const NodeInputParamterType& NodeInputParameter, const NodeWeightParamterType& NodeWeightParamter)
{
    /* Computation */
    float Output = 0.0f;
    for(int index = 0; index < NodeInputParameter.size(); index++)
    {
        Output += NodeInputParameter[index] * NodeWeightParamter[index];
    }
    //std::cout << " Before Sigmoid:" << Output;

    /* Activation */
    Output = 1 / ( 1 + exp(-Output)); 

    //std::cout << " After Sigmoid:" << Output;

    return Output;
}


float Network::getOutput(const InputParameterStruct& InputParams)
{
    /* hidden layer */
    //std::cout << "Age:" << InputParams.Age << " Class:" << InputParams.Class;
    NodeInputParamterType NodeInputParamter;
    NodeInputParamter.emplace_back(1.0f);
    NodeInputParamter.emplace_back(InputParams.Age);
    NodeInputParamter.emplace_back(InputParams.Class);
    NodeInputParamter.emplace_back(InputParams.Gender);

    NodeWeightParamterType NodeWeightParamter;
    NodeWeightParamter.emplace_back(mHLayerParameters[0][0]);
    NodeWeightParamter.emplace_back(mHLayerParameters[0][1]);
    NodeWeightParamter.emplace_back(mHLayerParameters[0][2]);
    NodeWeightParamter.emplace_back(mHLayerParameters[0][3]);
    float h1Node1Output = NodeOuput(NodeInputParamter, NodeWeightParamter);

    NodeWeightParamter.clear();
    NodeWeightParamter.emplace_back(mHLayerParameters[1][0]);
    NodeWeightParamter.emplace_back(mHLayerParameters[1][1]);
    NodeWeightParamter.emplace_back(mHLayerParameters[1][2]);
    NodeWeightParamter.emplace_back(mHLayerParameters[1][3]);
    float h1Node2Output = NodeOuput(NodeInputParamter, NodeWeightParamter);

    /* output layer */
    NodeInputParamter.clear();
    NodeInputParamter.emplace_back(1.0f);
    NodeInputParamter.emplace_back(h1Node1Output);
    NodeInputParamter.emplace_back(h1Node2Output);

    NodeWeightParamter.clear();
    NodeWeightParamter.emplace_back(mOutputLayerParamter[0]);
    NodeWeightParamter.emplace_back(mOutputLayerParamter[1]);
    NodeWeightParamter.emplace_back(mOutputLayerParamter[2]);
    
    float output = NodeOuput(NodeInputParamter, NodeWeightParamter);

    output = (output > 0.008f) ? 1.0f : 0.0f;
    //std::cout << " Output:" << output << "\n";
    return output;
}

#endif