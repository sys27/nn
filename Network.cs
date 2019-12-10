using System.Diagnostics;
using System.Text.Json;
using Troschuetz.Random.Distributions.Continuous;
using Troschuetz.Random.Generators;
using NN.ActivationFunctions;

namespace NN
{
    public class Network
    {
        private double[][][] weights;
        private double[][] biases;

        private readonly IActivationFunction activationFunction;

        public Network(params int[] neuronsCount)
        {
            this.activationFunction = new Sigmoid();

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

        public double[] FeedForward(double[][] layer, double[] layerBiases, double[] inputs)
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
                result = this.activationFunction.Activation(result);
                results[neuronIndex] = result;
            }

            return results;
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

        public double[][][] Weights => this.weights;

        public double[][] Biases => this.biases;

        public IActivationFunction ActivationFunction => this.activationFunction;
    }
}