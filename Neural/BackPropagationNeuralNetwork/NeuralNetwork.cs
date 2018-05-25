using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace BackPropagationNeuralNetwork
{
    public class NeuralNetwork
    {
        private int numInput;
        private int numHidden;
        private int numOutput;
        private static Random rnd;
        private double[] inputs;

        private double[][] ihWeights;
        private double[] hBiases;
        private double[] hOutputs;

        private double[][] hoWeights;
        private double[] oBiases;
        private double[] outputs;

        private double[] oGrads; // Output gradients for back-propagation
        private double[] hGrads; // Hidden gradients for back-propagation

        private double[][] ihPrevWeightsDelta; // For momentum with back-propagation
        private double[] hPrevBiasesDelta;
        private double[][] hoPrevWeightsDelta;
        private double[] oPrevBiasesDelta;

        /// <summary>
        /// n practical terms, except in rare situations, there are just three activation functions commonly
        /// used in neural networks: the hyperbolic tangent function, the logistic sigmoid function, and the
        /// softmax function. The calculus derivatives of these three functions at some value y are:
        /// hyperbolic tangent: (1 - y)(1 + y)
        /// logistic sigmoid: y(1 - y)
        /// softmax: y(1 - y)
        /// </summary>
        /// <param name="numInput"></param>
        /// <param name="numHidden"></param>
        /// <param name="numOutput"></param>
        public NeuralNetwork(int numInput, int numHidden, int numOutput)
        {
            this.numInput = numInput;
            this.numHidden = numHidden;
            this.numOutput = numOutput;

            this.inputs = new double[numInput];
            this.ihWeights = MakeMatrix(numInput, numHidden);
            this.hBiases = new double[numHidden];
            this.hOutputs = new double[numHidden];

            this.hoWeights = MakeMatrix(numHidden, numOutput);
            this.oBiases = new double[numOutput];
            this.outputs = new double[numOutput];

            oGrads = new double[numOutput];
            hGrads = new double[numHidden];

            ihPrevWeightsDelta = MakeMatrix(numInput, numHidden);
            hPrevBiasesDelta = new double[numHidden];
            hoPrevWeightsDelta = MakeMatrix(numHidden, numOutput);
            oPrevBiasesDelta = new double[numOutput];

            InitMatrix(ihPrevWeightsDelta, 0.011);
            InitVector(hPrevBiasesDelta, 0.011);
            InitMatrix(hoPrevWeightsDelta, 0.011);
            InitVector(oPrevBiasesDelta, 0.011);
            rnd = new Random(0);
        }

        private static double[][] MakeMatrix(int rows, int cols)
        {
            double[][] result = new double[rows][];
            for (int i = 0; i < rows; ++i)
            {
                result[i] = new double[cols];
            }
            return result;
        }

        private static void InitVector(double[] vector, double value)
        {
            for (int i = 0; i < vector.Length; i++)
                vector[i] = value;
        }

        private static void InitMatrix(double[][] matrix, double value)
        {
            int rows = matrix.Length;
            int cols = matrix[0].Length;
            for(int i = 0; i < rows; ++i)
                for(int j = 0; j < cols; ++j)
                    matrix[i][j] = value;
        }

        public void SetWeights(double[] weights)
        {
            int k = 0;

            for (int i = 0; i < numInput; ++i)
            {
                for (int j = 0; j < numHidden; ++j)
                {
                    ihWeights[i][j] = weights[k++];
                }
            }

            for (int i = 0; i < numHidden; ++i)
            {
                hBiases[i] = weights[k++];
            }

            for (int i = 0; i < numHidden; ++i)
            {
                for (int j = 0; j < numOutput; ++j)
                {
                    hoWeights[i][j] = weights[k++];
                }
            }

            for (int i = 0; i < numOutput; ++i)
            {
                oBiases[i] = weights[k++];
            }
        }

        public double[] GetWeights()
        {
            int numWeights = (numInput * numHidden) + numHidden + (numHidden * numOutput) + numOutput;
            double[] result = new double[numWeights];
            int k = 0; // Pointer into results array

            for (int i = 0; i < numInput; ++i)
            {
                for (int j = 0; j < numHidden; ++j)
                {
                    result[k++] = ihWeights[i][j];
                }
            }

            for (int i = 0; i < numHidden; ++i)
            {
                result[k++] = hBiases[i];
            }

            for (int i = 0; i < numHidden; ++i)
            {
                for (int j = 0; j < numOutput; ++j)
                {
                    result[k++] = hoWeights[i][j];
                }
            }

            for (int i = 0; i < numOutput; ++i)
            {
                result[k++] = oBiases[i];
            }
            return result;
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

        private static double HyperTan(double v)
        {
            if (v < -20.0)
                return -1.0;
            else if (v > 20.0)
                return 1.0;
            else
                return Math.Tanh(v);
        }

        private static double[] Softmax(double[] oSums)
        {
            double max = oSums[0];
            for (int i = 0; i < oSums.Length; ++i)
            {
                if (oSums[i] > max)
                    max = oSums[i];
            }

            double scale = 0.0;
            for (int i = 0; i < oSums.Length; ++i)
            {
                scale += Math.Exp(oSums[i] - max);
            }

            double[] result = new double[oSums.Length];
            for (int i = 0; i < oSums.Length; ++i)
            {
                result[i] = Math.Exp(oSums[i] - max) / scale;
            }

            return result;
        }

        public void FindWeights(double[] tValues, double[] xValues, double learnRate, double momentum, int maxEpochs)
        {
            int epoch = 0;
            while (epoch <= maxEpochs)
            {
                double[] yValues = ComputeOutputs(xValues);
                UpdateWeights(tValues, learnRate, momentum);
                if (epoch % 100 == 0)
                {
                    Console.WriteLine($"Epoch = {epoch.ToString().PadLeft(5)} Current outputs = ");
                    BackPropagationNeuralNetwork.Program.ShowVector(yValues, 2, 4, true);
                }
                ++epoch;
            }
        }

        private void UpdateWeights(double[] tValues, double learnRate, double momentum)
        {
            if (tValues.Length != numOutput)
                throw new Exception("target array now same length as output in UpdateWeights");

            //Recall that because the demo neural network uses softmax activation, the calculus derivative at
            //value y is y(1 - y)
            for (int i = 0; i < oGrads.Length; ++i)
            {
                double derivative = (1 - outputs[i]) * outputs[i];
                oGrads[i] = derivative * (tValues[i] - outputs[i]);
            }
            

            //Hidden gradients
            for (int i = 0; i < hGrads.Length; ++i)
            {
                double derivative = (1 - hOutputs[i]) * (1 + hOutputs[i]);
                double sum = 0.0;
                for (int j = 0; j < numOutput; ++j)
                {
                    sum += oGrads[j] * hoWeights[i][j];
                }
                hGrads[i] = derivative * sum;
            }

            //Input-to-hidden weights
            for (int i = 0; i < ihWeights.Length; ++i)
            {
                for (int j = 0; j < ihWeights[i].Length; ++j)
                {
                    double delta = learnRate * hGrads[j] * inputs[i];
                    ihWeights[i][j] += delta;
                    ihWeights[i][j] += momentum * ihPrevWeightsDelta[i][j];
                    ihPrevWeightsDelta[i][j] = delta; // Save the delta;
                }
            }

            //Update hidden biases
            for (int i = 0; i < hBiases.Length; ++i)
            {
                double delta = learnRate * hGrads[i];
                hBiases[i] += delta;
                hBiases[i] += momentum * hPrevBiasesDelta[i];
                hPrevBiasesDelta[i] = delta; // save delta;
            }

            //Hidden to output weights
            for (int i = 0; i < hoWeights.Length; i++)
            {
                for (int j = 0; j < hoWeights[i].Length; ++j)
                {
                    double delta = learnRate * oGrads[j] * hOutputs[i];
                    hoWeights[i][j] += delta;
                    hoWeights[i][j] += momentum * hoPrevWeightsDelta[i][j];
                    hoPrevWeightsDelta[i][j] = delta; //Save Delta;
                }
            }

            //Output node biases
            for (int i = 0; i < oBiases.Length; ++i)
            {
                double delta = learnRate * oGrads[i] * 1.0;
                oBiases[i] += delta;
                oBiases[i] += momentum * oPrevBiasesDelta[i];
                oPrevBiasesDelta[i] = delta; // Save delta
            }
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
                for(int j = 0; j < numOutput; ++j)
                {
                    double err = tValues[j] - yValues[j];
                    sumSquaredError += err * err;
                }
            }
            return sumSquaredError / trainData.Length;
        }

    }

}
