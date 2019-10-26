using System.Collections.Generic;
using NN.ActivationFunctions;

namespace NN
{
    public sealed class NetworkBuilder
    {
        private readonly List<Layer> layers;

        public NetworkBuilder()
        {
            layers = new List<Layer>();
        }

        public NetworkBuilder AddLayer(int neuronCount, int weightCount, IActivationFunction activationFunction)
        {
            var neurons = new Neuron[neuronCount];
            for (var i = 0; i < neuronCount; i++)
                neurons[i] = new Neuron(weightCount, activationFunction);

            layers.Add(new Layer(neurons));

            return this;
        }

        public Network Build()
        {
            return new Network(layers.ToArray());
        }
    }
}