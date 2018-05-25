using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkTraining
{
    class NeuralNetwork
    {
        private static Random rnd;
        private int numInput;
        private int numHidden;
        private int numOutput;

        public NeuralNetwork(int numInput, int numHidden, int numOutput) { }

        public void Train(double[][] trainData, int maxEpochs, double learnRate, double momentum)
        {
            int epoch = 0;
            double[] xValues = new double[numInput];
            double[] tValues = new double[numOutput]; // Target values

            int[] sequence = new int[trainData.Length];
            for (int i = 0; i < sequence.Length; ++i)
            {
                sequence[i] = i;
            }

            while (epoch < maxEpochs)
            {
                double mse = MeanSquaredError(trainData);
                Console.WriteLine(epoch + " " + mse.ToString("F4"));
                if (mse < 0.040) break;

                Shuffle(sequence);
                for (int i = 0; i < trainData.Length; ++i)
                {
                    int idx = sequence[i];
                    Array.Copy(trainData[idx], xValues, numInput);
                    Array.Copy(trainData[idx], numInput, tValues, 0, numOutput);
                    ComputeOutputs(xValues);
                    UpdateWeights(tValues, learnRate, momentum);
                }
                ++epoch;
            }

        }

        private double MeanCrossEntropyError(double[][] trainData)
        {
            double sumError = 0.0;
            double[] xValues = new double[numInput];
            double[] tValues = new double[numOutput];

            for (int i = 0; i < trainData.Length; ++i)
            {
                Array.Copy(trainData[i], xValues, numInput);
                Array.Copy(trainData[i], numInput, tValues, 0, numOutput);
                double[] yValues = this.ComputeOutputs(xValues);
                for (int j = 0; j < numOutput; ++j)
                {
                    sumError += Math.Log(yValues[j]) * tValues[j];
                }
            }
            return -1.0 * sumError / trainData.Length;

        }
        public double[] ComputeOutputs(double[] xValues)
        {
            double[] hSums = new double[numHidden];
            double[] oSums = new double[numOutput];

            for (int i = 0; i < xValues.Length; ++i)
            {
                inputs[i] = xValues[i];
            }

            for (int j = 0; j < numHidden; ++j)
            {
                for (int i = 0; i < numInput; ++i)
                {
                    hSums[j] += inputs[i] * ihWeights[i][j];
                }
            }

            for (int i = 0; i < numHidden; ++i)
            {
                hSums[i] += hBiases[i];
            }

            for (int i = 0; i < numHidden; ++i)
            {
                hOutputs[i] = HyperTan(hSums[i]);
            }

            for (int j = 0; j < numOutput; ++j)
            {
                for (int i = 0; i < numHidden; ++i)
                {
                    oSums[j] += hOutputs[i] * hoWeights[i][j];
                }
            }

            for (int i = 0; i < numOutput; ++i)
            {
                oSums[i] += oBiases[i];
            }

            double[] softOut = Softmax(oSums);
            for (int i = 0; i < outputs.Length; ++i)
            {
                outputs[i] = softOut[i];
            }

            double[] result = new double[numOutput];
            for (int i = 0; i < outputs.Length; ++i)
            {
                result[i] = outputs[i];
            }

            return result;

        }
        public double Accuracy(double[][] testData)
        {
            // Percentage correct using winner-takes all.
            int numCorrect = 0;
            int numWrong = 0;
            double[] xValues = new double[numInput];
            double[] tValues = new double[numOutput];
            double[] yValues;

            for (int i = 0; i < testData.Length; ++i)
            {
                Array.Copy(testData[i], xValues, numInput);
                Array.Copy(testData[i], numInput, tValues, 0, numOutput);
                yValues = this.ComputeOutputs(xValues);
                int maxIndex = MaxIndex(yValues);

                if (tValues[maxIndex] == 1.0)
                    ++numCorrect;
                else
                    ++numWrong;
            }

            return (numCorrect * 1.0) / (numCorrect + numWrong);

        }

        private static int MaxIndex(double[] vector)
        {
            int bigIndex = 0;
            double biggestVal = vector[0];
            for (int i = 0; i < vector.Length; ++i)
            {
                if (vector[i] > biggestVal)
                {
                    biggestVal = vector[i];
                    bigIndex = i;
                }
            }
            return bigIndex;
        }

        private static void Shuffle(int[] sequence)
        {
            for (int i = 0; i < sequence.Length; ++i)
            {
                int r = rnd.Next(i, sequence.Length);
                int tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }
        }
        private double MeanSquaredError(double[][] trainData)
        {

            // Average squared error per training item.
            double sumSquaredError = 0.0;
            double[] xValues = new double[numInput]; // First numInput values in trainData.
            double[] tValues = new double[numOutput]; // Last numOutput values.

            // Walk through each training case. Looks like (6.9 3.2 5.7 2.3) (0 0 1)
            for (int i = 0; i < trainData.Length; ++i)
            {
                Array.Copy(trainData[i], xValues, numInput);
                Array.Copy(trainData[i], numInput, tValues, 0, numOutput);
                double[] yValues = this.ComputeOutputs(xValues);
                for (int j = 0; j < numOutput; ++j)
                {
                    double err = tValues[j] - yValues[j];
                    sumSquaredError += err * err;
                }
            }
            return sumSquaredError / trainData.Length;
        }
    }
}
