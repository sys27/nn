namespace NN
{
    public struct TestSet
    {
        public double[] Inputs { get; }
        public double[] Outputs { get; }
        
        public TestSet(double[] inputs, double[] outputs)
        {
            Inputs = inputs;
            Outputs = outputs;
        }
    }
}