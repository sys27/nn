using System;
using System.Collections.Generic;
using System.Linq;

namespace NN
{
    public class Network
    {
        private Layer[] layers;

        public Network(Layer[] layers)
        {
            this.layers = layers;
        }

        public double[] Calculate(double[] inputs)
        {
            foreach (var layer in layers)
            {
                inputs = layer.Calculate(inputs);
            }

            return inputs;
        }

        public double Evaluate(TestSet[] testSets)
        {
            var error = 0.0; // TODO:

            foreach (var testSet in testSets)
            {
                var results = Calculate(testSet.Inputs);
                if (results.Length != testSet.Outputs.Length)
                    throw new Exception();

                error = results
                    .Zip(testSet.Outputs, (actual, expected) => Math.Pow(actual - expected, 2))
                    .Sum();
            }

            return error / testSets.Length;
        }
    }
}