namespace NN
{
    public struct TrainingSet
    {
        public double[] Inputs { get; }
        public double[] Outputs { get; }
        
        public TrainingSet(double[] inputs, double[] outputs)
        {
            Inputs = inputs;
            Outputs = outputs;
        }
    }
}