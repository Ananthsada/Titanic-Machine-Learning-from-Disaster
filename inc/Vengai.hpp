#ifndef __VENGAI_HPP__
#define __VENGAI_HPP__

#include <iostream>
#include <vector>
#include <chrono>

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
using LayerDeltaType = std::vector<float>;

enum class LayerType : uint16_t
{
    HiddenLayer = 0,
    OutputLayer
};

class Layer
{
    LayerType mLayerType;
    uint16_t mNodeCount;
    uint16_t mInputCount;
    uint16_t mLayerNumber;

    /**
     * @brief Input Weights list also contains biases.
     * 
     */
    LayerWeightsListType mLayerWeightsList;
    LayerWeightsListType mWeightsGradientList;
    LayerInputType mLayerInputList;
    LayerOutputType mLayerOutputList;
    LayerDeltaType mLayerDeltaList;

public:
    Layer(LayerType layerType, uint16_t LayerNumber, uint16_t numberOfNodes, uint16_t numberOfInputs);

    /**
     * @brief Set the Weights object. Biases are part of WeightsList
     * 
     * @param WeightsList 
     */
    LayerType getType() const;
    LayerDeltaType DeltaList() const;
    LayerWeightsListType WeightsList() const;
    void setWeights(const LayerWeightsListType& WeightsList);
    LayerOutputType forwardPropogate(const LayerInputType& input);
    void CalculateGradientOutputLayer();
    void CalculateGradientHiddenLayer(const Layer& NextLayer);
    void updateWeights();
};


Layer::Layer(LayerType layerType, uint16_t LayerNumber, uint16_t numberOfNodes, uint16_t numberOfInputs)
    : mLayerType(layerType), mNodeCount(numberOfNodes), mInputCount(numberOfInputs), mLayerNumber(LayerNumber)
{
    for(uint16_t NodeIndex = 0; NodeIndex < mNodeCount; NodeIndex++)
    {
        std::vector<float> WeightsList;
        WeightsList.emplace_back(0.5f);
        for(uint16_t InputIndex = 0; InputIndex < numberOfInputs; InputIndex++)
        {
            WeightsList.emplace_back(0.5f);
        }
        mLayerWeightsList.emplace_back(WeightsList);
    }
}

LayerType Layer::getType() const
{
    return mLayerType;
}

LayerDeltaType Layer::DeltaList() const
{
    return mLayerDeltaList;
}

LayerWeightsListType Layer::WeightsList() const
{
    return mLayerWeightsList;
}

void Layer::setWeights(const LayerWeightsListType& WeightsList)
{
    mLayerWeightsList = WeightsList;
}

LayerOutputType Layer::forwardPropogate(const LayerInputType& input)
{
    mLayerOutputList.clear();    
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
        //std::cout << "LinearCombination:" << LinearCombination << "\n";

        float output = Sigmoid(LinearCombination);
        //std::cout << "Output:" << output << " ";
        mLayerOutputList.emplace_back(output);
    }

    return mLayerOutputList;
}

/**
 * @brief Calculate delta for the current layer
 * 
 * @param input 
 */
void Layer::CalculateGradientOutputLayer()
{
    mWeightsGradientList.clear();
    mLayerDeltaList.clear();
    for(uint16_t WeightListIndex = 0; WeightListIndex < mLayerWeightsList.size(); WeightListIndex++)
    {
        /* Gradient for bias */
        std::vector<float> WeightsGradientList;
        float delta = SigmoidTransient(mLayerOutputList[WeightListIndex]);
        WeightsGradientList.emplace_back(delta);
        mLayerDeltaList.emplace_back(delta);

        /* Gradient for Weights */
        for(uint16_t InputIndex = 0; InputIndex < mLayerInputList.size(); InputIndex++)
        {
            float InnerDelta = delta * mLayerInputList[InputIndex];
            WeightsGradientList.emplace_back(InnerDelta);
        }
        mWeightsGradientList.emplace_back(WeightsGradientList);
    }
}

void Layer::CalculateGradientHiddenLayer(const Layer& NextLayer)
{
    mWeightsGradientList.clear();
    mLayerDeltaList.clear();

    auto NextLayerDeltaList = NextLayer.DeltaList();
    auto NextLayerWeightsList = NextLayer.WeightsList();

    for(uint16_t WeightListIndex = 0; WeightListIndex < mLayerWeightsList.size(); WeightListIndex++)
    {
        
        /* Gradient for bias */
        std::vector<float> WeightsGradientList;

        float delta = 0;
        for(uint16_t NextLayerWeightsIndex = 0; NextLayerWeightsIndex < NextLayerWeightsList.size(); NextLayerWeightsIndex++)
        {
            delta += NextLayerDeltaList[WeightListIndex] * NextLayerWeightsList[NextLayerWeightsIndex][WeightListIndex] * SigmoidTransient(mLayerOutputList[WeightListIndex]);    
        }
        WeightsGradientList.emplace_back(delta);
        mLayerDeltaList.emplace_back(delta);

        /* Gradient for Weights */
        for(uint16_t InputIndex = 0; InputIndex < mLayerInputList.size(); InputIndex++)
        {
            float InnerDelta = 0.0f;
            for(uint16_t NextLayerWeightsIndex = 0; NextLayerWeightsIndex < NextLayerWeightsList.size(); NextLayerWeightsIndex++)
            {
                InnerDelta = delta * mLayerInputList[InputIndex];
            }
            WeightsGradientList.emplace_back(InnerDelta);
        }
        mWeightsGradientList.emplace_back(WeightsGradientList);
    }
}

void Layer::updateWeights()
{
    LayerWeightsListType NewLayerWeightsList;
    for(uint16_t WeightListIndex = 0; WeightListIndex < mLayerWeightsList.size(); WeightListIndex++)
    {
        std::vector<float> NewWeights;
        for(uint16_t WeightsIndex = 0; WeightsIndex < mLayerWeightsList[WeightListIndex].size(); WeightsIndex++)
        {
            NewWeights.emplace_back(ScaleGradientInput(mLayerWeightsList[WeightListIndex][WeightsIndex], mWeightsGradientList[WeightListIndex][WeightsIndex]));
        }

        NewLayerWeightsList.emplace_back(NewWeights);
    }

    mLayerWeightsList.clear();
    mLayerWeightsList = NewLayerWeightsList;
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
            mLayersList.emplace_back(Layer(LayerType::HiddenLayer, layerIndex, *CurrNodeCountItr, inputCount));
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

    auto start = std::chrono::steady_clock::now();
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
                // std::cout << "LayerInput Size:" << LayerInput.size() << "\n";
                // std::cout << "{";
                // for(auto each : LayerInput)
                // {
                //     std::cout << each <<  " ";
                // }
                // std::cout << "}\n";
                LayerOutputType LayerOutput = eachLayer.forwardPropogate(LayerInput);
                // std::cout << "LayerOutput Size:" << LayerOutput.size() << "\n";
                // std::cout << "{";
                // for(auto each : LayerOutput)
                // {
                //     std::cout << each <<  " ";
                // }
                // std::cout << "}\n";
                LayerInput.clear();
                LayerInput = LayerOutput;
            }


            auto eachLayerItr = mLayersList.rbegin();
            auto PrevLayerItr = mLayersList.rbegin();
            eachLayerItr->CalculateGradientOutputLayer();
            eachLayerItr++;

            for(;eachLayerItr != mLayersList.rend(); eachLayerItr++, PrevLayerItr++)
            {
                eachLayerItr->CalculateGradientHiddenLayer(*PrevLayerItr);
            }

            //std::cout << "\n";

            float SumOutput = 0;
            for(auto eachOutput : LayerInput)
            {
                SumOutput += eachOutput;
            }

            CurrentCalculatedCost += square(SumOutput - output[InputIndex]);
        }

        CurrentCalculatedCost = CurrentCalculatedCost / ( 2 * input.size());
        std::cout << "Previous Cost:" << PreviousCalculatedCost << " Current Cost:" << CurrentCalculatedCost << "\n";
        if(fabs(PreviousCalculatedCost - CurrentCalculatedCost) <= 0.0000001f)
                break;

        PreviousCalculatedCost = CurrentCalculatedCost;
        
        /* backward propogation for each Layer */
        for(auto& eachLayer : mLayersList)
        {
            eachLayer.updateWeights();
        }
    }

    float elaspedTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    std::cout << "Elapsed(us)=" << elaspedTime << std::endl;
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
    return (SumOutput > 0.003f) ? 1.0f : 0.0f;
}

}
#endif