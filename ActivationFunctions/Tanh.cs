using System;

namespace NN.ActivationFunctions
{
    public class Tanh : IActivationFunction
    {
        public double Activation(double value) => Math.Tanh(value);

        public double Derivative(double value) => 1 - Math.Pow(value, 2);
    }
}