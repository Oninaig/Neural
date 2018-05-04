using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    /// <summary>
    /// Matrix member ihWeights holds the weights from input nodes to hidden nodes, where the row
    /// index corresponds to the index of an input node, and the column index corresponds to the index
    /// of a hidden node. The matrix is implemented as an array of arrays. Unlike most programming
    /// languages, C# has a true multidimensional array and you may want to use that approach to
    /// store the neural network weights.
    /// Member array hBiases holds the bias values for the hidden nodes. Many implementations you'll
    /// find will omit this array and instead treat the hidden node biases as extra input-to-hidden
    /// weights.
    /// Member array hOutputs stores the hidden node outputs after summing the products of weights
    /// and inputs, adding the bias value, and applying an activation function during the computation of
    /// the output values. An alternative is to make this array local to the ComputeOutputs. However,
    /// because in most situations the ComputeOutputs method is called many thousands of times, a
    /// local array would have to be allocated many times. Naming array hOutputs is a bit tricky
    /// because the values also serve as inputs to the output layer.
    /// Member matrix hoWeights holds the weights from hidden nodes to output nodes. The row index
    /// of the matrix corresponds to the index of a hidden node and the column index corresponds to
    /// the index of an output node.
    /// Member array oBiases holds bias values for the output nodes. The member array named
    /// outputs holds the final overall computed neural network output values. As with the inputs array,
    /// you'll see that the outputs array can potentially be dropped from the design.
    /// 
    /// IMPORTANT NOTE ON ACTIVATION FUNCTIONS!
    /// Although there are some exceptions, in general the hyperbolic tangent function is the best
    /// choice for hidden layer activation. For output layer activation, if your neural network is
    /// performing classification where the dependent variable to be predicted has three or more values
    /// (for example, predicting a person's political inclination which can be "liberal", "moderate", or
    /// "conservative"), softmax activation is the best choice. If your neural network is performing
    /// classification where the dependent variable has exactly two possible values (for example,
    /// predicting a person's gender which can be "male" or "female"), the logistic sigmoid activation
    /// function is the best choice for output layer activation. 
    /// </summary>
    public class ForwardFeedNerualNetwork
    {
        private int numInput;
        private int numHidden;
        private int numOutput;

        private double[] inputs;
        private double[][] ihWeights;
        private double[] hBiases;
        private double[] hOutputs;

        private double[][] hoWeights;
        private double[] oBiases;

        private double[] outputs;

        //Example data weights
        ///double[] weights = new double[] {
        ///  0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
        ///  0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20,
        ///  0.21, 0.22, 0.23, 0.24, 0.25, 0.26
        /// };
        public ForwardFeedNerualNetwork(int numInput, int numHidden, int numOutput)
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

        }

        private static double[][] MakeMatrix(int rows, int cols)
        {
            double[][] result = new double[rows][];
            for (int i = 0; i < rows; ++i)
                result[i] = new double[cols];
            return result;
        }

        /// <summary>
        /// The method assumes that the values in the weights array parameter are stored in a particular
        /// order: first the input-to-hidden weights, followed by the hidden biases, followed by the hidden-tooutput
        /// weights, followed by the output biases. Additionally, the values for the two weights
        /// matrices are assumed to be stored in row-major order. This means the values are ordered from
        /// left to right and top to bottom.
        /// </summary>
        /// <param name="weights"></param>
        public void SetWeights(double[] weights)
        {
            int numWeights = (numInput * numHidden) + numHidden + (numHidden * numOutput) + numOutput;
            if (weights.Length != numWeights)
                throw new Exception("Bad weights array");

            int k = 0; // Pointer into weights parameter;

            for (int i = 0; i < numInput; ++i)
                for (int j = 0; j < numHidden; ++j)
                    ihWeights[i][j] = weights[k++];

            for (int i = 0; i < numHidden; ++i)
                hBiases[i] = weights[k++];

            for (int i = 0; i < numHidden; ++i)
                for (int j = 0; j < numOutput; ++j)
                    hoWeights[i][j] = weights[k++];

            for (int i = 0; i < numOutput; ++i)
                oBiases[i] = weights[k++];
        }

        public double[] ComputeOutputs(double[] xValues)
        {
            if (xValues.Length != numInput)
                throw new Exception("Bad xValues array");

            double[] hSums = new double[numHidden];
            double[] oSums = new double[numOutput];

            for (int i = 0; i < xValues.Length; ++i)
                inputs[i] = xValues[i];

            for (int j = 0; j < numHidden; ++j)
                for (int i = 0; i < numInput; ++i)
                    hSums[j] += inputs[i] * ihWeights[i][j];

            for (int i = 0; i < numHidden; ++i)
                hSums[i] += hBiases[i];

            Console.WriteLine("\nPre-activation hidden sums:");
            ForwardFeedNeuralNetworkProgram.ShowVector(hSums, 4, 4, true);


            for (int i = 0; i < numHidden; ++i)
                hOutputs[i] = HyperTan(hSums[i]);

            Console.WriteLine("\nHidden outputs:");
            ForwardFeedNeuralNetworkProgram.ShowVector(hOutputs, 4, 4, true);

            for (int j = 0; j < numOutput; ++j)
                for (int i = 0; i < numHidden; ++i)
                    oSums[j] += hOutputs[i] * hoWeights[i][j];

            for (int i = 0; i < numOutput; ++i)
                oSums[i] += oBiases[i];

            Console.WriteLine("\nPre-activation output sums:");
            ForwardFeedNeuralNetworkProgram.ShowVector(oSums, 2, 4, true);

            double[] softOut = Softmax(oSums); // Softmax does all outputs at once

            for (int i = 0; i < outputs.Length; ++i)
                outputs[i] = softOut[i];

            double[] result = new double[numOutput];
            for (int i = 0; i < outputs.Length; ++i)
                result[i] = outputs[i];

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

        public static double LogSigmoid(double x)
        {
            if (x < -45.0)
                return 0.0;
            else if (x > 45.0)
                return 1.0;
            else
                return 1.0 / (1.0 + Math.Exp(-x));
        }


        /// <summary>
        /// The mathematical definition of the softmax function is a bit difficult to express, so an example
        /// may be the best way to explain. Suppose there are three values, 1.0, 4.0, and 2.0.
        /// softmax(1.0) = e^1.0 / (e^1.0 + e^4.0 + e^2.0) = 2.7183 / (2.7183 + 54.5982 + 7.3891) = 0.04
        /// softmax(4.0) = e^4.0 / (e^1.0 + e^4.0 + e^2.0) = 54.5982 / (2.7183 + 54.5982 + 7.3891) = 0.84
        /// softmax(2.0) = e^2.0 / (e^1.0 + e^4.0 + e^2.0) = 7.3891 / (2.7183 + 54.5982 + 7.3891) = 0.12
        /// The problem with the naive implementation of the softmax function is that the denominator term
        /// can easily get very large or very small and cause a potential arithmetic overflow. It is possible to
        /// implement a more sophisticated version of the softmax function by using some clever math. 
        /// </summary>
        /// <param name="oSums"></param>
        /// <returns></returns>
        private static double[] NaiveSoftmax(double[] oSums)
        {
            double denom = 0.0;
            for (int i = 0; i < oSums.Length; ++i)
                denom += Math.Exp(oSums[i]);

            double[] result = new double[oSums.Length];
            for (int i = 0; i < oSums.Length; ++i)
                result[i] = Math.Exp(oSums[i]) / denom;
            return result;
        }

        /// <summary>
        /// Using the three values from the previous (naive) example, 1.0, 4.0, and 2.0, the first step is to
        /// determine the largest value, which in this case is max = 4.0. The next step is to compute a
        /// scaling factor which is the sum of e raised to each input value minus the maximum input value:
        /// scale = e^(1.0 - max) + e^(4.0 - max) + e^(2.0 - max) = e^-3.0 + e^0^+ e^-2.0 = 0.0498 + 1 + 0.1353 = 1.1851
        /// softmax(1.0) = e^(1.0 - max) / scale = e^-3.0 / 1.1851 = 0 .0498 / 1.1851 = 0.04
        /// softmax(4.0) = e^(4.0 - max) / scale = e^0^/ 1.1851 = 1 / 1.1851 = 0.84
        /// softmax(2.0) = e^(2.0 - max) / scale = e^-2.0 / 1.1851 = 0 .1353 / 1.1851 = 0.12
        /// </summary>
        /// <param name="oSums"></param>
        /// <returns></returns>
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
                scale += Math.Exp(oSums[i] - max);

            double[] result = new double[oSums.Length];
            for (int i = 0; i < oSums.Length; ++i)
                result[i] = Math.Exp(oSums[i] - max) / scale;

            return result; // Cell values sum to ~1.0
        }

    }




}
