using System;
using System.Collections.Generic;
using System.Linq;
using NN.Data;

namespace NN.LearningAlgorithms
{
    public class BackpropagationAlgorithm : ILearningAlgorithm
    {
        private readonly Network network;

        private const double learningRate = 1.0;

        public BackpropagationAlgorithm(Network network)
        {
            this.network = network;
        }

        public void Train(IEnumerable<DataSet> trainingSets)
        {
            var random = new Random();
            foreach (var trainSet in trainingSets.OrderBy(x => random.Next()))
                Train(trainSet);
        }

        public double Train(IEnumerable<DataSet> trainingSets, IEnumerable<DataSet> testSets)
        {
            Train(trainingSets);

            return Evaluate(testSets);
        }

        public void Train(DataSet trainingSet)
        {
            var weights = this.network.Weights;
            var biases = this.network.Biases;
            var activationFunction = this.network.ActivationFunction;

            var activations = new double[weights.Length + 1][];
            activations[0] = trainingSet.Inputs;

            for (var layerIndex = 0; layerIndex < weights.Length; layerIndex++)
            {
                var layer = weights[layerIndex];
                var layerBiases = biases[layerIndex];
                var inputs = activations[layerIndex];

                activations[layerIndex + 1] = this.network.FeedForward(layer, layerBiases, inputs);
            }

            var deltaWeights = new double[weights.Length][][];
            var deltaBiases = new double[biases.Length][];

            // delta for last layer
            var lastLayerActivation = activations[^1];
            var delta = new double[lastLayerActivation.Length];
            for (var activationIndex = 0; activationIndex < lastLayerActivation.Length; activationIndex++)
            {
                var activation = lastLayerActivation[activationIndex];
                delta[activationIndex] = (activation - trainingSet.Outputs[activationIndex]) * activationFunction.Derivative(activation);
            }

            deltaBiases[^1] = delta;
            deltaWeights[^1] = new double[delta.Length][];
            for (var deltaIndex = 0; deltaIndex < delta.Length; deltaIndex++)
            {
                var activation = activations[^2];
                deltaWeights[^1][deltaIndex] = new double[activation.Length];
                for (var activationIndex = 0; activationIndex < activation.Length; activationIndex++)
                {
                    deltaWeights[^1][deltaIndex][activationIndex] = activation[activationIndex] * delta[deltaIndex];
                }
            }

            // delta for all layers except last
            for (var layerIndex = weights.Length - 2; layerIndex >= 0; layerIndex--)
            {
                var layerWeight = weights[layerIndex + 1];

                var newDelta = new double[activations[layerIndex + 1].Length];
                for (var deltaIndex = 0; deltaIndex < newDelta.Length; deltaIndex++)
                {
                    var error = 0.0;
                    for (var neuronIndex = 0; neuronIndex < layerWeight.Length; neuronIndex++)
                    {
                        error += layerWeight[neuronIndex][deltaIndex] * delta[neuronIndex];
                    }

                    var deriv = activationFunction.Derivative(activations[layerIndex + 1][deltaIndex]);

                    newDelta[deltaIndex] = error * deriv;
                }
                delta = newDelta;

                deltaBiases[layerIndex] = delta;
                deltaWeights[layerIndex] = new double[delta.Length][];
                for (var deltaIndex = 0; deltaIndex < delta.Length; deltaIndex++)
                {
                    var activation = activations[layerIndex];
                    deltaWeights[layerIndex][deltaIndex] = new double[activation.Length];
                    for (var activationIndex = 0; activationIndex < activation.Length; activationIndex++)
                    {
                        deltaWeights[layerIndex][deltaIndex][activationIndex] = activation[activationIndex] * delta[deltaIndex];
                    }
                }
            }

            UpdateBiases(deltaBiases);
            UpdateWeights(deltaWeights);
        }

        public double Evaluate(IEnumerable<DataSet> testSets)
        {
            var error = testSets
                .AsParallel()
                .Select(Evaluate)
                .Average();

            return error;
        }

        public double Evaluate(DataSet testSet)
        {
            var mse = 0.0;
            var results = this.network.Calculate(testSet.Inputs);

            for (var i = 0; i < results.Length; i++)
                mse += (results[i] - testSet.Outputs[i]) * (results[i] - testSet.Outputs[i]);

            mse /= results.Length;

            return Math.Sqrt(mse);
        }

        private void UpdateBiases(double[][] deltaBiases)
        {
            var biases = this.network.Biases;

            for (var layerIndex = 0; layerIndex < biases.Length; layerIndex++)
            {
                var layerBias = biases[layerIndex];
                for (var biasIndex = 0; biasIndex < layerBias.Length; biasIndex++)
                {
                    layerBias[biasIndex] -= learningRate * deltaBiases[layerIndex][biasIndex];
                }
            }
        }

        private void UpdateWeights(double[][][] deltaWeights)
        {
            var weights = this.network.Weights;

            for (var layerIndex = 0; layerIndex < weights.Length; layerIndex++)
            {
                var layer = weights[layerIndex];

                for (var neuronIndex = 0; neuronIndex < layer.Length; neuronIndex++)
                {
                    var neuron = layer[neuronIndex];

                    for (var weightIndex = 0; weightIndex < neuron.Length; weightIndex++)
                    {
                        neuron[weightIndex] -= learningRate * deltaWeights[layerIndex][neuronIndex][weightIndex];
                    }
                }
            }
        }
    }
}