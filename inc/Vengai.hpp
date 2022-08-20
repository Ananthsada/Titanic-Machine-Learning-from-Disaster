#ifndef __VENGAI_HPP__
#define __VENGAI_HPP__

#include <iostream>
#include <vector>

namespace Vengai
{

constexpr float GRADIENT_SCALE_FACTOR = 1.5f;

constexpr float square(float input)
{
    return fabs(input) * fabs(input);
}

/**
 * @brief Calculate Sigmoid of function
 * 
 * @param input 
 * @return float 
 */
float Sigmoid(const float input)
{
    return 1 / (1 + exp(-input));
}

float ScaleGradientInput(float input, float gradient)
{
    return input - (GRADIENT_SCALE_FACTOR * gradient);
}

/**
 * @brief Calculate Gradient of Sigmoid function
 * 
 * @param input 
 * @return float 
 */
float SigmoidTransient(const float input)
{
    return input * (1 - input);
}

using LayerWeightsListType = std::vector<std::vector<float>>;
using LayerInputType = std::vector<float>;
using LayerOutputType = std::vector<float>;

class Layer
{
    uint16_t mNodeCount;
    uint16_t mInputCount;
    uint16_t mLayerNumber;

    /**
     * @brief Input Weights list also contains biases.
     * 
     */
    LayerWeightsListType mLayerWeightsList;
    LayerWeightsListType mAccumulatedWeightList;
    uint16_t mAccumulationCount;
    LayerInputType mLayerInputList;

public:
    Layer(uint16_t LayerNumber, uint16_t numberOfNodes, uint16_t numberOfInputs);

    /**
     * @brief Set the Weights object. Biases are part of WeightsList
     * 
     * @param WeightsList 
     */
    void setWeights(const LayerWeightsListType& WeightsList);
    LayerOutputType forwardPropogate(const LayerInputType& input);
    LayerWeightsListType getChangeinWeights(const LayerInputType& input);
    void addWeights(const LayerWeightsListType& WeightsList);
    void backPropogate();
};


Layer::Layer(uint16_t LayerNumber, uint16_t numberOfNodes, uint16_t numberOfInputs)
    : mNodeCount(numberOfNodes), mInputCount(numberOfInputs), mLayerNumber(LayerNumber)
{
    for(uint16_t NodeIndex = 0; NodeIndex < mNodeCount; NodeIndex++)
    {
        std::vector<float> WeightsList;
        WeightsList.emplace_back(1.0f);
        for(uint16_t InputIndex = 0; InputIndex < numberOfInputs; InputIndex++)
        {
            WeightsList.emplace_back(1.0f);
        }
        mLayerWeightsList.emplace_back(WeightsList);
    }
}

void Layer::setWeights(const LayerWeightsListType& WeightsList)
{
    mLayerWeightsList = WeightsList;
}

LayerOutputType Layer::forwardPropogate(const LayerInputType& input)
{
    LayerOutputType LayerOutput;
    
    mLayerInputList = input;
    // std::cout << "{";
    // for(auto each : mLayerInputList)
    // {
    //     std::cout << each << " ";
    // }
    // std::cout << "}\n";
    
    for(uint16_t NodeIndex = 0; NodeIndex < mNodeCount; NodeIndex++)
    {
        //std::cout << "WeightList size:" << mLayerWeightsList[NodeIndex].size() << "\n";
        std::vector<float> NodeWeight = mLayerWeightsList[NodeIndex];

        float LinearCombination = NodeWeight[0];
        for(size_t InputIndex = 0; InputIndex < mLayerInputList.size(); InputIndex++)
        {
            LinearCombination += mLayerInputList[InputIndex] * NodeWeight[InputIndex + 1];
        }

        float output = Sigmoid(LinearCombination);
        //std::cout << "Output:" << output << " ";
        LayerOutput.emplace_back(output);
    }

    return LayerOutput;
}

LayerWeightsListType Layer::getChangeinWeights(const LayerInputType& input)
{
    LayerWeightsListType TotalWeightsList;

    for(auto eachNodeWeights : mLayerWeightsList)
    {
        std::vector<float> WeightsList;
        float change = ScaleGradientInput(eachNodeWeights[0], 1.0f);
        std::cout << "Layer:" << mLayerNumber << " Current Weight:" << eachNodeWeights[0] << " Changed:" << change << "\n";
        WeightsList.emplace_back(change);
        for(size_t InputIndex = 0; InputIndex < mLayerInputList.size(); InputIndex++)
        {
            change = ScaleGradientInput(eachNodeWeights[InputIndex + 1], mLayerInputList[InputIndex]);
            WeightsList.emplace_back(change);
            std::cout << "Layer:" << mLayerNumber << " Input:" << mLayerInputList[InputIndex] << "\n";
            std::cout << "Layer:" << mLayerNumber << " Current Weight:" << eachNodeWeights[InputIndex + 1] << " Changed:" << change << "\n";
        }

        TotalWeightsList.emplace_back(WeightsList);
    }

    return TotalWeightsList;
}

void Layer::addWeights(const LayerWeightsListType& WeightsList)
{
    LayerWeightsListType LocalList;
    for(uint16_t NodeIndex = 0; NodeIndex < WeightsList.size(); NodeIndex++)
    {
        std::vector<float> list;
        for(uint16_t WeightIndex = 0; WeightIndex < WeightsList[NodeIndex].size(); WeightIndex++)
        {
            float value = 0;
            if(mAccumulatedWeightList.size() == 0)
            {
                value = WeightsList[NodeIndex][WeightIndex];
            }
            else
            {
                value = (mAccumulatedWeightList[NodeIndex][WeightIndex] * mAccumulationCount) + WeightsList[NodeIndex][WeightIndex];
                value = (value / (mAccumulationCount + 1));
            }
            list.emplace_back(value);
        }
        LocalList.emplace_back(list);
    }
    mAccumulationCount++;
    mAccumulatedWeightList.clear();
    mAccumulatedWeightList = LocalList;
}

void Layer::backPropogate()
{
    mLayerWeightsList = mAccumulatedWeightList;
}

struct NetworkConfig
{
    uint16_t mLayerCount;
    uint16_t mInputCount;
    uint16_t mOutputCount;
    std::vector<uint16_t> mNodeCount;
};

using NetworkInputType = std::vector<std::vector<float>>;
using NetworkOutputType = std::vector<float>;

class Network
{
    NetworkConfig mNetworkConfig;
    std::vector<Layer> mLayersList;

    void initLayers();
public:
    Network(const NetworkConfig& networkConfig);

    void train(const NetworkInputType& input, const NetworkOutputType& output);
    float test(const std::vector<float>& input);
};


Network::Network(const NetworkConfig& networkConfig)
{
    mNetworkConfig = networkConfig;

    initLayers();
}

void Network::initLayers()
{
    auto CurrNodeCountItr = mNetworkConfig.mNodeCount.begin();
    uint16_t inputCount = mNetworkConfig.mInputCount;
    for(uint16_t layerIndex = 0; layerIndex < mNetworkConfig.mLayerCount; layerIndex++)
    {
        if(CurrNodeCountItr != mNetworkConfig.mNodeCount.end())
        {
            mLayersList.emplace_back(Layer(layerIndex, *CurrNodeCountItr, inputCount));
            inputCount = *CurrNodeCountItr;
            CurrNodeCountItr++;
        }
    }
}

void Network::train(const NetworkInputType& input, const NetworkOutputType& output)
{
    /* Run until finding local minimum of cost function */
    float CurrentCalculatedCost = 0;
    float PreviousCalculatedCost = 0;

    std::cout << "Total No of inputs:" << input.size() << "\n";
    // std::cout << "{";
    // for(auto each : input)
    // {
    //     std::cout << each.size() <<  " ";
    // }
    // std::cout << "}\n";
    while(true)
    {
        CurrentCalculatedCost = 0;
        /* forward propogation for each input */
        for(size_t InputIndex = 0; InputIndex < input.size(); InputIndex++)
        {
            /* Layer by Layer */
            LayerInputType LayerInput = input[InputIndex];
            //std::cout << "Current size:" << LayerInput.size() << "\n";
            for(auto& eachLayer : mLayersList)
            {
                std::cout << "LayerInput Size:" << LayerInput.size() << "\n";
                std::cout << "{";
                for(auto each : LayerInput)
                {
                    std::cout << each <<  " ";
                }
                std::cout << "}\n";
                LayerOutputType LayerOutput = eachLayer.forwardPropogate(LayerInput);
                // std::cout << "LayerOutput Size:" << LayerOutput.size() << "\n";
                // std::cout << "{";
                // for(auto each : LayerOutput)
                // {
                //     std::cout << each <<  " ";
                // }
                // std::cout << "}\n";
                LayerWeightsListType LayerWeightsList = eachLayer.getChangeinWeights(LayerInput);
                eachLayer.addWeights(LayerWeightsList);
                LayerInput.clear();
                LayerInput = LayerOutput;
            }
            std::cout << "\n";

            float SumOutput = 0;
            for(auto eachOutput : LayerInput)
            {
                SumOutput += eachOutput;
            }

            CurrentCalculatedCost += square(SumOutput - output[InputIndex]);
        }

        CurrentCalculatedCost = CurrentCalculatedCost / ( 2 * input.size());
        std::cout << "Previous Cost:" << PreviousCalculatedCost << " Current Cost:" << CurrentCalculatedCost << "\n";
        if(fabs(PreviousCalculatedCost - CurrentCalculatedCost) <= 0.0001f)
                break;

        PreviousCalculatedCost = CurrentCalculatedCost;
        
        /* backward propogation for each Layer */
        for(auto& eachLayer : mLayersList)
        {
            eachLayer.backPropogate();
        }
    }
}


float Network::test(const std::vector<float>& input)
{
    /* Run until finding local minimum of cost function */
    float CurrentCalculatedCost = 0;
    float PreviousCalculatedCost = 0;

    LayerInputType LayerInput;
    LayerOutputType LayerOutput;

    for(auto& eachLayer : mLayersList)
    {
        LayerOutputType LayerOutput = eachLayer.forwardPropogate(input);

        LayerInput.clear();
        LayerInput = LayerOutput;
    }

    float SumOutput = 0;
    for(auto eachOutput : LayerInput)
    {
        SumOutput += eachOutput;
    }

    std::cout << "Sumoutput:" << SumOutput << "\n";
    return (SumOutput > 0.00003f) ? 1.0f : 0.0f;
}

}
#endif