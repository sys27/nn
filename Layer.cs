using System.Linq;

namespace NN
{
    public class Layer
    {
        private readonly Neuron[] neurons;

        public Layer(Neuron[] neurons)
        {
            this.neurons = neurons;
        }

        public double[] Calculate(double[] inputs)
        {
            return neurons
                .Select(neuron => neuron.Calculate(inputs))
                .ToArray();
        }
    }
}