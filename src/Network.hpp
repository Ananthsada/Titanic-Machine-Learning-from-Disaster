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
};

using InputParameterMapType = std::map<uint16_t, InputParameterStruct>;
using OutputParameterMapType = std::map<uint16_t, float>;

constexpr uint8_t LAYER_COUNT = 1;
constexpr uint8_t HLAYER_NODE_COUNT = 2;

class Network
{
public:
    Network();
    ~Network() {}

    void Train(const InputParameterMapType& InputParameterMap, const OutputParameterMapType& OutputParamterMap);
private:
    using NodeInputParamterType = std::vector<float>;
    using NodeWeightParamterType = std::vector<float>;
    using NodeBiasParamterType = std::vector<float>;

    float mHLayerParameters[HLAYER_NODE_COUNT][3];
    float mOutputLayerParamter[3];
    OutputParameterMapType mOutputParamter;
    float NodeOuput(const NodeInputParamterType& NodeInputParameter, const NodeWeightParamterType& NodeWeightParamter);
    float ForwardPropogation(const InputParameterStruct& InputParams);
};

Network::Network()
{
    mHLayerParameters[0][0] = 1.0f;
    mHLayerParameters[0][1] = 1.0f;
    mHLayerParameters[0][2] = 1.0f;
    mHLayerParameters[1][0] = 1.0f;
    mHLayerParameters[1][1] = 1.0f;
    mHLayerParameters[1][2] = 1.0f;

    mOutputLayerParamter[0] = 1.0f;
    mOutputLayerParamter[1] = 1.0f;
}

void Network::Train(const InputParameterMapType& InputParameter, const OutputParameterMapType& OutputParamter)
{
    auto start = std::chrono::steady_clock::now();
    std::cout << "Input Paramter Size:" << InputParameter.size() << "\n";
    for(const auto& each : InputParameter)
    {
        float out = ForwardPropogation(each.second);
        mOutputParamter.insert({each.first, out});
    }
    std::cout << "Output Paramter Size:" << mOutputParamter.size() << "\n";

    float elaspedTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    std::cout << "Elapsed(ms)=" << elaspedTime << std::endl;
}

float Network::ForwardPropogation(const InputParameterStruct& InputParams)
{
    /* hidden layer */
    NodeInputParamterType NodeInputParamter;
    NodeInputParamter.emplace_back(1.0f);
    NodeInputParamter.emplace_back(InputParams.Age);
    NodeInputParamter.emplace_back(InputParams.Class);

    NodeWeightParamterType NodeWeightParamter;
    NodeWeightParamter.emplace_back(mHLayerParameters[0][0]);
    NodeWeightParamter.emplace_back(mHLayerParameters[0][1]);
    NodeWeightParamter.emplace_back(mHLayerParameters[0][2]);
    float h1Node1Output = NodeOuput(NodeInputParamter, NodeWeightParamter);

    NodeWeightParamter.clear();
    NodeWeightParamter.emplace_back(mHLayerParameters[1][0]);
    NodeWeightParamter.emplace_back(mHLayerParameters[1][1]);
    NodeWeightParamter.emplace_back(mHLayerParameters[1][2]);
    float h1Node2Output = NodeOuput(NodeInputParamter, NodeWeightParamter);

    /* output layer */
    NodeInputParamter.clear();
    NodeInputParamter.emplace_back(1.0f);
    NodeInputParamter.emplace_back(h1Node1Output);
    NodeInputParamter.emplace_back(h1Node2Output);

    NodeWeightParamter.clear();
    NodeWeightParamter.emplace_back(mOutputLayerParamter[0]);
    NodeWeightParamter.emplace_back(mOutputLayerParamter[1]);
    
    return NodeOuput(NodeInputParamter, NodeWeightParamter);
}

float Network::NodeOuput(const NodeInputParamterType& NodeInputParameter, const NodeWeightParamterType& NodeWeightParamter)
{
    /* Computation */
    float Output = 0.0f;
    for(int index = 0; index < NodeInputParameter.size(); index++)
    {
        Output += NodeInputParameter[index] * NodeWeightParamter[index];
    }

    /* Activation */
    return 1 / ( 1 + exp(-Output)); 
}

#endif