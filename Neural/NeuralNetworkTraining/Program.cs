using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BackPropagationNeuralNetwork;

namespace NeuralNetworkTraining
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("\nBegin neural network training demo");
            double[][] allData = new double[150][];
            string dataFile = "IrisData.txt";
            allData = LoadData(dataFile, 150, 7);

            Console.WriteLine("\nFirst 6 rows of the 150-item data set:");
            ShowMatrix(allData, 6, 1, true);

            Console.WriteLine("Creating 80% training and 20% test data matrices");
            double[][] trainData = null;
            double[][] testData = null;
            MakeTrainTest(allData, 72, out trainData, out testData);


            Console.WriteLine("\nFirst 3 rows of training data:");
            ShowMatrix(trainData, 3, 1, true);
            Console.WriteLine("First 3 rows of test data:");
            ShowMatrix(testData, 3, 1, true);

            Console.WriteLine("\nCreating a 4-input, 7-hidden, 3-output neural network");
            Console.Write("Hard-coded tanh function for input-to-hidden and softmax for ");
            Console.WriteLine("hidden-to-output activations");

            //The number of input nodes and the number of output nodes are determined by the structure of
            //the source data: four numeric x - values and a y - value with three categorical values. Specifying a
            //good value for the number of hidden nodes is one of the major challenges when working with
            //neural networks.Even though there has been much research done in this area, picking a good
            //value for the number of hidden nodes is mostly a matter of trial and error.
            int numInput = 4;
            int numHidden = 7;
            int numOutput = 3;
            NeuralNetwork nn = new NeuralNetwork(numInput, numHidden, numOutput);


            int maxEpochs = 1000;
            double learnRate = 0.05;
            double momentum = 0.01;
            Console.WriteLine("Setting maxEpochs = " + maxEpochs + ", learnRate = " +
                              learnRate + ", momentum = " + momentum);
            Console.WriteLine("Training has hard-coded mean squared " +
                              "error < 0.040 stopping condition");
            Console.WriteLine("\nBeginning training using incremental back-propagation\n");
            nn.Train(trainData, maxEpochs, learnRate, momentum);
            Console.WriteLine("Training complete");


            double[] weights = nn.GetWeights();
            Console.WriteLine("Final neural network weights and bias values:");
            ShowVector(weights, 10, 3, true);

            double trainAcc = nn.Accuracy(trainData);
            Console.WriteLine("\nAccuracy on training data = " + trainAcc.ToString("F4"));
            double testAcc = nn.Accuracy(testData);
            Console.WriteLine("\nAccuracy on test data = " + testAcc.ToString("F4"));
            Console.WriteLine("\nEnd neural network training demo\n");
            Console.ReadLine()

        }
        static void ShowVector(double[] vector, int valsPerRow, int decimals, bool newLine)
        {
            for (int i = 0; i < vector.Length; ++i)
            {
                if (i % valsPerRow == 0) Console.WriteLine("");
                Console.Write(vector[i].ToString("F" + decimals).PadLeft(decimals + 4) + " ");
            }
            if (newLine == true)
                Console.WriteLine("");
        }
        static double[][] LoadData(string dataFile, int numRows, int numCols)
        {
            double[][] result = new double[numRows][];
            FileStream ifs = new FileStream(dataFile, FileMode.Open);
            StreamReader sr = new StreamReader(ifs);
            string line = "";
            string[] tokens = null;
            int i = 0;
            while ((line = sr.ReadLine()) != null)
            {
                tokens = line.Split(',');
                result[i] = new double[numCols];
                for (int j = 0; j < numCols; ++j)
                {
                    result[i][j] = double.Parse(tokens[j]);
                }
                ++i;
            }
            sr.Close();
            ifs.Close();
            return result;
        }
        static void MakeTrainTest(double[][] allData, int seed,
            out double[][] trainData, out double[][] testData)
        {
            Random rnd = new Random(seed);
            int totRows = allData.Length;
            int numCols = allData[0].Length;
            int trainRows = (int) (totRows * 0.80); // Hard-coded 80-20 split
            int testRows = totRows = trainRows;
            trainData = new double[trainRows][];
            testData = new double[testRows][];

            double[][] copy = new double[allData.Length][];
            for (int i = 0; i < copy.Length; ++i)
                copy[i] = allData[i];

            for (int i = 0; i < copy.Length; ++i)
            {
                int r = rnd.Next(i, copy.Length);
                double[] tmp = copy[r];
                copy[r] = copy[i];
                copy[i] = tmp;
            }

            for (int i = 0; i < trainRows; ++i)
            {
                trainData[i] = new double[numCols];
                for (int j = 0; j < numCols; ++j)
                {
                    trainData[i][j] = copy[i][j];
                }
            }

            for (int i = 0; i < testRows; ++i)
            {
                testData[i] = new double[numCols];
                for (int j = 0; j < numCols; ++j)
                {
                    testData[i][j] = copy[i + trainRows][j]; // be careful
                }
            }
        }
    

        static void ShowMatrix(double[][] matrix, int numRows, int decimals, bool newLine)
        {
            for (int i = 0; i < numRows; ++i)
            {
                Console.Write(i.ToString().PadLeft(3) + ": ");
                for (int j = 0; j < matrix[i].Length; ++j)
                {
                    if (matrix[i][j] >= 0.0) Console.Write(" "); else Console.Write("-");
                    Console.Write(Math.Abs(matrix[i][j]).ToString("F" + decimals) + " ");
                }
                Console.WriteLine("");
            }
            if (newLine == true)
                Console.WriteLine("");
        }

    }
}
