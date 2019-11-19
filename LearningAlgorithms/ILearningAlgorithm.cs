using System.Collections.Generic;
using NN.Data;

namespace NN.LearningAlgorithms
{
    public interface ILearningAlgorithm
    {
        void Train(IEnumerable<DataSet> trainingSets);
        void Train(DataSet trainingSet);
        double Train(IEnumerable<DataSet> trainingSets, IEnumerable<DataSet> testSets);

        double Evaluate(IEnumerable<DataSet> testSets);
        double Evaluate(DataSet testSet);
    }
}