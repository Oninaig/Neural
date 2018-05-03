using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace Perceptron
{
    public class Perceptron
    {
        private int numInput;
        private double[] inputs;
        private double[] weights;
        private double bias;
        private int output;
        private Random rnd;

        public Perceptron(int numInput)
        {
            this.numInput = numInput;
            this.inputs = new double[numInput];
            this.weights = new double[numInput];
            this.rnd = new Random(0); //Initializing with 0 (hard-coded) allows us to reproduce training runs
            InitializeWeights();
        }

        private void InitializeWeights()
        {
            double lo = -0.01;
            double hi = 0.01;
            for (int i = 0; i < weights.Length; i++)
                weights[i] = (hi - lo) * rnd.NextDouble() + lo;
            bias = (hi - lo) * rnd.NextDouble() + lo;
        }

        public int ComputeOutput(double[] xValues)
        {
            if (xValues.Length != numInput)
                throw new Exception("Bad xValues in ComputeOutput");
            for (int i = 0; i < xValues.Length; ++i)
            {
                this.inputs[i] = xValues[i];
            }
            double sum = 0.0;
            for (int i = 0; i < numInput; ++i)
            {
                sum += this.inputs[i] * this.weights[i];
            }
            sum += this.bias;
            int result = Activation(sum);
            this.output = result;
            return result;
        }
        private static int Activation(double v)
        {
            if (v >= 0.0)
                return +1;
            else
                return -1;
        }

        public double[] Train(double[][] trainData, double alpha, int maxEpochs)
        {
            int epoch = 0;
            double[] xValues = new double[numInput];
            int desired = 0;

            int[] sequence = new int[trainData.Length];
            for (int i = 0; i < sequence.Length; ++i)
            {
                sequence[i] = i;
            }

            while (epoch < maxEpochs)
            {
                Shuffle(sequence);
                for (int i = 0; i < trainData.Length; i++)
                {
                    int idx = sequence[i];
                    Array.Copy(trainData[idx], xValues, numInput);
                    desired = (int) trainData[idx][numInput]; // -1 or +1
                    int computed = ComputeOutput(xValues);
                    Update(computed, desired, alpha); // Modify weights and bias values;
                }
                ++epoch;
            }
            double[] result = new double[numInput + 1];
            Array.Copy(this.weights, result, numInput);
            result[result.Length - 1] = bias; // Last cell
            return result;
        }

        private void Shuffle(int[] sequence)
        {
            for (int i = 0; i < sequence.Length; i++)
            {
                int r = rnd.Next(i, sequence.Length);
                int tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }
        }

        /// <summary>
        /// Method Update calculates the difference between the computed output and the desired output
        /// and stores the difference into the variable delta. Delta will be positive if the computed output is
        /// too large, or negative if computed output is too small. For a perceptron with -1 and +1 outputs,
        /// delta will always be either -2 (if computed = -1 and desired = +1), or +2 (if computed = +1 and
        /// desired = -1), or 0 (if computed equals desired).
        /// 
        /// For each weight[i], if the computed output is too large, the weight is reduced by amount (alpha *
        /// delta * input[i]). If input[i] is positive, the product term will also be positive because alpha and
        /// delta are also positive, and so the product term is subtracted from weight[i]. If input[i] is
        /// negative, the product term will be negative, and so to reduce weight[i] the product term must be
        /// added.
        /// 
        /// Notice that the size of the change in a weight is proportional to both the magnitude of delta and
        /// the magnitude of the weight's associated input value. So a larger delta produces a larger
        /// change in weight, and a larger associated input also produces a larger weight change.
        /// The learning rate alpha scales the magnitude of a weight change. Larger values of alpha
        /// generate larger changes in weight which leads to faster learning, but at a risk of overshooting a
        /// good weight value. Smaller values of alpha avoid overshooting but make training slower.
        /// </summary>
        /// <param name="computed"></param>
        /// <param name="desired"></param>
        /// <param name="alpha"></param>
        private void Update(int computed, int desired, double alpha)
        {
            if (computed == desired) return; // we're good;
            int delta = computed - desired; // if computed > desired, delta is +. In our demo it can either be -2, 0, or +2

            for (int i = 0; i < this.weights.Length; ++i) // Each input-weight pair
            {
                if (computed > desired && inputs[i] >= 0.0) // need to reduce weights;
                    weights[i] = weights[i] - (alpha * delta * inputs[i]); // delta is +, input is +
                else if (computed > desired && inputs[i] < 0.0) // need to reduce weights
                    weights[i] = weights[i] + (alpha * delta * inputs[i]); // delta is +, input is -
                else if (computed < desired && inputs[i] >= 0.0) // need to increase weights
                    weights[i] = weights[i] - (alpha * delta * inputs[i]); // delta is -, input is +
                else if (computed < desired && inputs[i] < 0.0) // need to increase weights
                    weights[i] = weights[i] + (alpha * delta * inputs[i]); //delta is -, input is -

            } // each weight
            bias = bias - (alpha * delta);


        }
    }
}
