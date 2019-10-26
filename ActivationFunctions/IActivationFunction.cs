namespace NN.ActivationFunctions
{
    public interface IActivationFunction
    {
        double Activation(double value);
        double Derivative(double value);
    }
}