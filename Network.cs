using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text.Json;
using Troschuetz.Random.Distributions.Continuous;
using Troschuetz.Random.Generators;

namespace NN
{
    public class Network
    {
        private double[][][] weights;

        private double[][] biases;

        private const double learningRate = 1.0;

        public Network(params int[] neuronsCount)
        {
            var random = new NormalDistribution(new StandardGenerator(), 0, 1);

            weights = new double[neuronsCount.Length - 1][][];
            biases = new double[neuronsCount.Length - 1][];

            for (var i = 1; i < neuronsCount.Length; i++)
            {
                var index = i - 1;

                weights[index] = new double[neuronsCount[i]][];
                for (var neuronIndex = 0; neuronIndex < weights[index].Length; neuronIndex++)
                {
                    weights[index][neuronIndex] = new double[neuronsCount[index]];
                    for (var weightIndex = 0; weightIndex < weights[index][neuronIndex].Length; weightIndex++)
                        weights[index][neuronIndex][weightIndex] = random.NextDouble();
                }

                biases[index] = new double[neuronsCount[i]];
                for (var biasIndex = 0; biasIndex < biases[index].Length; biasIndex++)
                    biases[index][biasIndex] = random.NextDouble();
            }
        }

        public double[] Calculate(double[] inputs)
        {
            for (var layerIndex = 0; layerIndex < weights.Length; layerIndex++)
            {
                var layer = weights[layerIndex];
                var layerBiases = biases[layerIndex];

                inputs = FeedForward(layer, layerBiases, inputs);
            }

            return inputs;
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
            var results = Calculate(testSet.Inputs);

            for (var i = 0; i < results.Length; i++)
                mse += (results[i] - testSet.Outputs[i]) * (results[i] - testSet.Outputs[i]);

            mse /= results.Length;

            return Math.Sqrt(mse);
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
            var activations = new double[weights.Length + 1][];
            activations[0] = trainingSet.Inputs;

            for (var layerIndex = 0; layerIndex < weights.Length; layerIndex++)
            {
                var layer = weights[layerIndex];
                var layerBiases = biases[layerIndex];
                var inputs = activations[layerIndex];

                activations[layerIndex + 1] = FeedForward(layer, layerBiases, inputs);
            }

            var deltaWeights = new double[weights.Length][][];
            var deltaBiases = new double[biases.Length][];

            // delta for last layer
            var lastLayerActivation = activations[^1];
            var delta = new double[lastLayerActivation.Length];
            for (var activationIndex = 0; activationIndex < lastLayerActivation.Length; activationIndex++)
            {
                var activation = lastLayerActivation[activationIndex];
                delta[activationIndex] = (activation - trainingSet.Outputs[activationIndex]) * Derivative(activation);
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

                    var deriv = Derivative(activations[layerIndex + 1][deltaIndex]);

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

        public string Save()
        {
            var configuration = new Configuration { Weights = this.weights, Biases = this.biases };

            return JsonSerializer.Serialize(configuration);
        }

        public void Load(string json)
        {
            var configuration = JsonSerializer.Deserialize<Configuration>(json);

            this.weights = configuration.Weights;
            this.biases = configuration.Biases;
        }

        private void UpdateBiases(double[][] deltaBiases)
        {
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

        private double Activation(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        private double Derivative(double x)
        {
            return x * (1.0 - x);
        }

        private double[] FeedForward(double[][] layer, double[] layerBiases, double[] inputs)
        {
            var results = new double[layer.Length];

            for (var neuronIndex = 0; neuronIndex < layer.Length; neuronIndex++)
            {
                var neuron = layer[neuronIndex];
                var bias = layerBiases[neuronIndex];

                Debug.Assert(neuron.Length == inputs.Length);

                var result = 0.0;
                for (var weightIndex = 0; weightIndex < neuron.Length; weightIndex++)
                {
                    result += neuron[weightIndex] * inputs[weightIndex];
                }

                result += bias;
                result = Activation(result);
                results[neuronIndex] = result;
            }

            return results;
        }
    }
}