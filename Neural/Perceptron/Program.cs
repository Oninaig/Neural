using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Perceptron
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("\nBegin perceptron demo\n");
            Console.WriteLine("Predict liberal (-1) or conservative (+1) from age, income");

            double[][] trainData = new double[8][];
            trainData[0] = new double[] { 1.5, 2.0, -1 };
            trainData[1] = new double[] { 2.0, 3.5, -1 };
            trainData[2] = new double[] { 3.0, 5.0, -1 };
            trainData[3] = new double[] { 3.5, 2.5, -1 };
            trainData[4] = new double[] { 4.5, 5.0, 1 };
            trainData[5] = new double[] { 5.0, 7.0, 1 };
            trainData[6] = new double[] { 5.5, 8.0, 1 };
            trainData[7] = new double[] { 6.0, 6.0, 1 };

            Console.WriteLine("\nThe training data is:\n");
            ShowData(trainData);

            Console.WriteLine("\nCreating perceptron");
            int numInput = 2;
            Perceptron p = new Perceptron(numInput);
            double alpha = 0.001;
            int maxEpochs = 100;

            Console.Write("\nSetting learning rate to " + alpha.ToString("F3"));
            Console.WriteLine(" and maxEpochs to " + maxEpochs);

            Console.WriteLine("\nBegin training");
            double[] weights = p.Train(trainData, alpha, maxEpochs);
            Console.WriteLine("Training complete");

            Console.WriteLine("\nBest weights and bias found:");
            ShowVector(weights, 4, true);

            double[][] newData = new double[6][];
            newData[0] = new double[] { 3.0, 4.0 }; // Should be -1.
            newData[1] = new double[] { 0.0, 1.0 }; // Should be -1.
            newData[2] = new double[] { 2.0, 5.0 }; // Should be -1.
            newData[3] = new double[] { 5.0, 6.0 }; // Should be 1.
            newData[4] = new double[] { 9.0, 9.0 }; // Should be 1.
            newData[5] = new double[] { 4.0, 6.0 }; // Should be 1.

            Console.WriteLine("\nPredictions for new people:\n");
            for (int i = 0; i < newData.Length; ++i)
            {
                Console.Write("Age, Income = ");
                ShowVector(newData[i], 1, false);
                int c = p.ComputeOutput(newData[i]);
                Console.Write(" Prediction is ");
                if (c == -1)
                    Console.WriteLine("(-1) liberal");
                else if (c == 1)
                    Console.WriteLine("(+1) conservative");
            }

            Console.WriteLine("\nEnd perceptron demo\n");
            Console.ReadLine();
        }

        static void ShowData(double[][] trainData)
        {
            int numRows = trainData.Length;
            int numCols = trainData[0].Length;
            for (int i = 0; i < numRows; ++i)
            {
                Console.Write("[" + i.ToString().PadLeft(2, ' ') + "] ");
                for (int j = 0; j < numCols - 1; ++j)
                    Console.Write(trainData[i][j].ToString("F1").PadLeft(6));
                Console.WriteLine(" -> " + trainData[i][numCols - 1].ToString("+0;-0"));
            }
        }

        static void ShowVector(double[] vector, int decimals, bool newLine)
        {
            for (int i = 0; i < vector.Length; i++)
            {
                if (vector[i] >= 0.0)
                {
                   Console.Write(" "); // For sign.
                }
                Console.Write(vector[i].ToString("F" + decimals) + " ");
            }

            if(newLine == true)
                Console.WriteLine("");
        }


    }
}
