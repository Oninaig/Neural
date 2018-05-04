using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BackPropagationNeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
        }

        public static void ShowVector(double[] vector, int valsPerRow, int decimals,
            bool newLine)
        {
            for (int i = 0; i < vector.Length; ++i)
            {
                if (i > 0 && i % valsPerRow == 0)
                    Console.WriteLine("");
                Console.Write(vector[i].ToString("F" + decimals).PadLeft(decimals + 4) + " ");
            }
            if (newLine == true) Console.WriteLine("");
        }
        public static void ShowMatrix(double[][] matrix, int decimals)
        {
            int cols = matrix[0].Length;
            for (int i = 0; i < matrix.Length; ++i) // Each row.
                ShowVector(matrix[i], cols, decimals, true);
        }

    }
}
