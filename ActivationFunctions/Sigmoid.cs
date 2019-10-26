using System;

namespace NN.ActivationFunctions
{
    public class Sigmoid : IActivationFunction
    {
        public double Activation(double value) => 1 / (1 + Math.Exp(-value));

        public double Derivative(double value) => value * (1 - value);
    }
}