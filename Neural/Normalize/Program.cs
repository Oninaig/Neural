using System;

namespace Normalize
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            Console.WriteLine("\nBegin data encoding and normalization demo\n");
            string[] sourceData =
            {
                "Sex Age Locale Income Politics",
                "==============================================",
                "Male 25 Rural 63,000.00 Conservative",
                "Female 36 Suburban 55,000.00 Liberal",
                "Male 40 Urban 74,000.00 Moderate",
                "Female 23 Rural 28,000.00 Liberal"
            };
            Normalizer.ShowData(sourceData);

            string[] encodedData = new string[] {
                "-1 25 1 0 63,000.00 1 0 0",
                " 1 36 0 1 55,000.00 0 1 0",
                "-1 40 -1 -1 74,000.00 0 0 1",
                " 1 23 1 0 28,000.00 0 1 0" };

            Console.WriteLine("\nData after categorical encoding:\n");
            Normalizer.ShowData(encodedData);

            Console.WriteLine("\nNumeric data stored in matrix:\n");
            double[][] numericData = new double[4][];
            numericData[0] = new double[] { -1, 25.0, 1, 0, 63000.00, 1, 0, 0 };
            numericData[1] = new double[] { 1, 36.0, 0, 1, 55000.00, 0, 1, 0 };
           
            numericData[2] = new double[] { -1, 40.0, -1, -1, 74000.00, 0, 0, 1 };
            numericData[3] = new double[] { 1, 23.0, 1, 0, 28000.00, 0, 1, 0 };
            Normalizer.ShowMatrix(numericData, 2);

            Normalizer.GuassNormal(numericData, 1);
            Normalizer.MinMaxNormal(numericData, 4);

            Console.WriteLine("\nMatrix after normalization (Gaussian col. 1" +
                              " and MinMax col. 4):\n");
            Normalizer.ShowMatrix(numericData, 2);
            Console.WriteLine("\nEnd data encoding and normalization demo\n");
            Console.ReadLine();

        }
    }
}