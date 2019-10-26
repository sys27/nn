namespace NN.ActivationFunctions
{
    public static class Functions
    {
        private static IActivationFunction tanh = new Tanh();
        private static IActivationFunction sigmoid = new Sigmoid();

        public static IActivationFunction Tanh => tanh;
        public static IActivationFunction Sigmoid => sigmoid;
    }
}