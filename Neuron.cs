using System;
using System.Linq;
using NN.ActivationFunctions;

namespace NN
{
    public class Neuron
    {
        private readonly IActivationFunction activationFunction;

        private double[] weights;

        private double bias;

        public Neuron(int weightCount, IActivationFunction activationFunction)
        {
            this.activationFunction = activationFunction;
            var random = new Random();

            weights = new double[weightCount];
            for (var i = 0; i < weightCount; i++)
            {
                weights[i] = random.NextDouble();
            }

            bias = random.NextDouble();
        }

        public double Calculate(double[] inputs)
        {
            var result = weights
                .Zip(inputs, (weight, input) => weight * input)
                .Sum();

            return activationFunction.Activation(result + bias);
        }
    }
}