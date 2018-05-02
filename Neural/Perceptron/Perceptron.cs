using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Perceptron
{
    public class Perceptron
    {
        private int numInput;
        private double[] input;
        private double[] weights;
        private double bias;
        private int output;
        private Random rnd;

        public Perceptron(int numInput) { }
        private void InitializeWeights() { }
        public int ComputeOutput(double[] xValues)
        {
            throw new NotImplementedException();
        }
        private static int Activation(double v)
        {
            throw new NotImplementedException();
        }

        public double[] Train(double[][] trainData, double alpha, int maxEpochs)
        {
            throw new NotImplementedException();
        }
        private void Shuffle(int[] sequence) { }
        private void Update(int computed, int desired, double alpha) { }
    }
}
