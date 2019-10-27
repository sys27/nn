namespace NN
{
    public struct DataSet
    {
        public double[] Inputs { get; }
        public double[] Outputs { get; }
        
        public DataSet(double[] inputs, double[] outputs)
        {
            Inputs = inputs;
            Outputs = outputs;
        }
    }
}