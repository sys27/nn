using System;
using System.Diagnostics;
using System.Threading.Tasks;
using NN.Data;
using NN.LearningAlgorithms;

namespace NN
{
    class Program
    {
        private const string TrainFile = "mnist_train.csv";
        private const string TestFile = "mnist_test.csv";

        static async Task Main(string[] args)
        {
            var csvReader = new CsvReader();

            var trainSets = await csvReader.Read(TrainFile);
            var testSets = await csvReader.Read(TestFile);

            var nn = new Network(784, 30, 10);
            var bp = new BackpropagationAlgorithm(nn);

            var error = 0.0;
            var count = 0;
            do
            {
                count++;

                var sw = Stopwatch.StartNew();

                error = bp.Train(trainSets, testSets);

                sw.Stop();
                Console.WriteLine(sw.ElapsedMilliseconds);

                Console.WriteLine($"Count: {count} - MSE: {error}.");
            } while (error > 0.005);

            Console.WriteLine(nn.Save());

            Console.ReadLine();
        }
    }
}