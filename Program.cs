using System;
using NN.ActivationFunctions;

namespace NN
{
    class Program
    {
        static void Main(string[] args)
        {
            var nn = new NetworkBuilder()
                .AddLayer(2, 2, Functions.Sigmoid)
                .AddLayer(1, 2, Functions.Sigmoid)
                .Build();

            Console.WriteLine("Hello World!");
        }
    }
}