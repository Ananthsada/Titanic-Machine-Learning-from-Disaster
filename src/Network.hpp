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
        for(const auto& each : InputParameter)
        {
            float out = ForwardPropogation(each.second);

            CurrentCost += sqrt(OutputParamter[index] - out);
            index++;
            std::cout << out << " ";
        }
        std::cout << "\n";

        CurrentCost = CurrentCost / ( 2 * index);

        if((PrevCost - CurrentCost) < 0.001f)
        {
            break;
        }

        float delta = SigmoidTransient(mOutputLayerParamter[0] + (mOutputLayerParamter[1] * ));
        float delH1 = 0.0f;
        float delH2 = 0.0f;
        mHLayerParameters[0][0] = mHLayerParameters[0][0] - (SCALE_FACTOR * );
        mHLayerParameters[0][1] = mHLayerParameters[0][1] - ;
        mHLayerParameters[0][2] = mHLayerParameters[0][2] - ;
        mHLayerParameters[1][0] = mHLayerParameters[1][0] - ;
        mHLayerParameters[1][1] = mHLayerParameters[1][1] - ;
        mHLayerParameters[1][2] = mHLayerParameters[1][2] - ;

        mOutputLayerParamter[0] = mOutputLayerParamter[0] - ;
        mOutputLayerParamter[1] = mOutputLayerParamter[1] - ;
        mOutputLayerParamter[2] = mOutputLayerParamter[2] - ;
        
        PrevCost = CurrentCost;
    }

    float elaspedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
    std::cout << "Elapsed(us)=" << elaspedTime << std::endl;
}

float Network::ForwardPropogation(const InputParameterStruct& InputParams)
{
    /* hidden layer */
    //std::cout << "Age:" << InputParams.Age << " Class:" << InputParams.Class;
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
    NodeWeightParamter.emplace_back(mOutputLayerParamter[2]);
    
    float output = NodeOuput(NodeInputParamter, NodeWeightParamter);

    //std::cout << " Output:" << output << "\n";
    return output;
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

#endif